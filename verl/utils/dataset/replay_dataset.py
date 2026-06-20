# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import glob
import json
import os
import re
from typing import List, Optional

from torch.utils.data import Dataset
from transformers import PreTrainedTokenizer

from verl.utils.model import compute_position_id_with_mask
import verl.utils.torch_functional as verl_F


def _load_jsonl_records(jsonl_path: str) -> List[dict]:
    records = []
    with open(jsonl_path, encoding='utf-8') as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                records.append(json.loads(line))
            except json.JSONDecodeError as exc:
                raise ValueError(f'Invalid JSON on line {line_no} of {jsonl_path}: {exc}') from exc
    return records


def _step_sort_key(path: str) -> int:
    match = re.search(r'step_(\d+)\.jsonl$', os.path.basename(path))
    if match is None:
        raise ValueError(f'Unexpected step replay filename (expected step_NNNN.jsonl): {path}')
    return int(match.group(1))


def _load_replay_records(path: str) -> List[dict]:
    if os.path.isfile(path):
        return _load_jsonl_records(path)

    if not os.path.isdir(path):
        raise FileNotFoundError(f'Replay path not found (file or directory): {path}')

    step_files = sorted(glob.glob(os.path.join(path, 'step_*.jsonl')), key=_step_sort_key)
    if not step_files:
        raise FileNotFoundError(f'No step_*.jsonl files found in replay directory: {path}')

    records = []
    for step_file in step_files:
        file_records = _load_jsonl_records(step_file)
        print(f'loaded {len(file_records)} replay records from {step_file}')
        records.extend(file_records)
    print(f'loaded {len(records)} replay records total from {path}')
    return records


def _dedupe_records(
    records: List[dict],
    dedupe_key: str,
    dedupe_strategy: str,
) -> List[dict]:
    if dedupe_key == 'none':
        return records

    grouped = {}
    for record in records:
        if dedupe_key not in record:
            raise KeyError(
                f'dedupe_key={dedupe_key!r} missing in replay record id={record.get("id")!r}'
            )
        key = record[dedupe_key]
        if key not in grouped:
            grouped[key] = record
            continue
        existing = grouped[key]
        if dedupe_strategy == 'latest_step':
            existing_step = existing.get('step', -1)
            new_step = record.get('step', -1)
            if new_step >= existing_step:
                grouped[key] = record
        elif dedupe_strategy == 'first':
            pass
        else:
            raise ValueError(f'Unknown dedupe_strategy: {dedupe_strategy!r}')
    return list(grouped.values())


def _reconstruct_student_chat(question: str) -> list:
    think_open = '<' + 'redacted_thinking' + '>'
    think_close = '</' + 'redacted_thinking' + '>'
    prefix = (
        "Answer the given question. "
        f"You must conduct reasoning inside {think_open} and {think_close} first every time you get new information. "
        "After reasoning, if you find you lack some knowledge, you can call a search engine by <search> query </search> "
        "and it will return the top searched results between <information> and </information>. "
        "You can search as many times as your want. "
        "If you find no further external knowledge needed, you can directly provide the answer inside <answer> and </answer>, "
        f"without detailed illustrations. For example, <answer> Beijing </answer>. Question: {question}\n"
    )
    return [{'role': 'user', 'content': prefix}]


def _coerce_chat(value) -> Optional[list]:
    if value is None:
        return None
    if isinstance(value, list) and value:
        return value
    return None


def _resolve_chat(record: dict, prompt_mode: str) -> list:
    if prompt_mode == 'teacher':
        chat = _coerce_chat(record.get('teacher_prompt'))
        if chat is None:
            raise ValueError(
                f'Replay record id={record.get("id")!r} missing teacher_prompt; '
                f'use eval_prompt_mode=student or a train_replay.jsonl file.'
            )
        return chat

    chat = _coerce_chat(record.get('prompt'))
    if chat is None:
        chat = _coerce_chat(record.get('raw_prompt'))
    if chat is None:
        question = record.get('question')
        if not question:
            raise ValueError(
                f'Replay record id={record.get("id")!r} has no prompt/raw_prompt/question '
                f'for student-mode eval.'
            )
        chat = _reconstruct_student_chat(question)
    return chat


def _filter_distillation_active(records: List[dict], jsonl_path: str) -> List[dict]:
    if not records:
        return records
    if 'self_distillation_mask' not in records[0]:
        raise ValueError(
            f'eval_replay_require_distillation_mask=true requires self_distillation_mask '
            f'in replay records, but {jsonl_path} has none (e.g. val_replay). '
            f'Use train_replay.jsonl or disable the filter.'
        )
    filtered = [r for r in records if float(r.get('self_distillation_mask', 0)) > 0]
    return filtered


def _validate_teacher_prompts(records: List[dict], jsonl_path: str) -> None:
    missing = [
        r.get('id', r.get('index'))
        for r in records
        if _coerce_chat(r.get('teacher_prompt')) is None
    ]
    if missing:
        raise ValueError(
            f'eval_prompt_mode=teacher requires teacher_prompt in every replay record, '
            f'but {len(missing)} record(s) in {jsonl_path} lack it '
            f'(e.g. val replay). Use eval_prompt_mode=student or a train_replay.jsonl file.'
        )


class ReplayJsonlDataset(Dataset):
    """Dataset that loads eval samples from train/val replay JSONL exports."""

    def __init__(
        self,
        jsonl_path: str,
        tokenizer: PreTrainedTokenizer,
        prompt_mode: str = 'student',
        max_prompt_length: int = 4096,
        dedupe_key: str = 'index',
        dedupe_strategy: str = 'latest_step',
        replay_limit: Optional[int] = None,
        truncation: Optional[str] = None,
        require_distillation_mask: bool = False,
    ):
        if prompt_mode not in ('student', 'teacher'):
            raise ValueError(f'prompt_mode must be student or teacher, got {prompt_mode!r}')
        if dedupe_key not in ('index', 'id', 'none'):
            raise ValueError(f'dedupe_key must be index, id, or none, got {dedupe_key!r}')
        if dedupe_strategy not in ('latest_step', 'first'):
            raise ValueError(
                f'dedupe_strategy must be latest_step or first, got {dedupe_strategy!r}'
            )

        if not os.path.isfile(jsonl_path) and not os.path.isdir(jsonl_path):
            raise FileNotFoundError(f'Replay path not found (file or directory): {jsonl_path}')

        self.jsonl_path = jsonl_path
        self.tokenizer = tokenizer
        self.prompt_mode = prompt_mode
        self.max_prompt_length = max_prompt_length
        self.truncation = truncation or ('right' if prompt_mode == 'teacher' else 'error')
        self.truncated_count = 0
        self.require_distillation_mask = require_distillation_mask

        raw_records = _load_replay_records(jsonl_path)
        if os.path.isfile(jsonl_path):
            print(f'loaded {len(raw_records)} replay records from {jsonl_path}')

        if require_distillation_mask:
            raw_records = _filter_distillation_active(raw_records, jsonl_path)
            print(f'after distillation mask filter (self_distillation_mask > 0): {len(raw_records)} records')
            if not raw_records:
                raise ValueError(
                    f'No replay records with self_distillation_mask > 0 in {jsonl_path}'
                )

        if prompt_mode == 'teacher':
            _validate_teacher_prompts(raw_records, jsonl_path)

        records = _dedupe_records(raw_records, dedupe_key, dedupe_strategy)
        print(
            f'after dedupe (key={dedupe_key}, strategy={dedupe_strategy}): '
            f'{len(records)} records'
        )

        if replay_limit is not None:
            records = records[:replay_limit]
            print(f'after replay_limit={replay_limit}: {len(records)} records')

        self.records = records

    def __len__(self):
        return len(self.records)

    def __getitem__(self, item):
        record = self.records[item]
        chat = _resolve_chat(record, self.prompt_mode)

        if self.tokenizer.chat_template:
            prompt_with_chat_template = self.tokenizer.apply_chat_template(
                chat, add_generation_prompt=True, tokenize=False
            )
        else:
            prompt_with_chat_template = chat[0]['content']

        pre_trunc_len = len(
            self.tokenizer.encode(prompt_with_chat_template, add_special_tokens=False)
        )
        input_ids, attention_mask = verl_F.tokenize_and_postprocess_data(
            prompt=prompt_with_chat_template,
            tokenizer=self.tokenizer,
            max_length=self.max_prompt_length,
            pad_token_id=self.tokenizer.pad_token_id,
            left_pad=True,
            truncation=self.truncation,
        )
        post_trunc_len = int(attention_mask[0].sum().item())
        if post_trunc_len < pre_trunc_len:
            self.truncated_count += 1

        position_ids = compute_position_id_with_mask(attention_mask)

        row_dict = {
            'input_ids': input_ids[0],
            'attention_mask': attention_mask[0],
            'position_ids': position_ids[0],
            'raw_prompt': chat,
            'prompt': chat,
            'reward_model': {'ground_truth': record['ground_truth']},
            'data_source': record.get('data_source', 'unknown'),
            'question': record.get('question'),
            'index': record.get('index'),
            'id': record.get('id'),
        }
        for optional_key in ('uid', 'step', 'self_distillation_mask'):
            if optional_key in record:
                row_dict[optional_key] = record[optional_key]
        return row_dict
