import torch
from torch import nn

from verl.utils.torch_functional import ema_update_module_


def test_ema_update_module_blends_student_into_teacher():
    teacher = nn.Linear(4, 2)
    student = nn.Linear(4, 2)
    update_rate = 0.1

    teacher_before = {name: p.clone() for name, p in teacher.named_parameters()}
    student_params = {name: p.clone() for name, p in student.named_parameters()}

    ema_update_module_(teacher, student, update_rate)

    for name, t_param in teacher.named_parameters():
        expected = (1.0 - update_rate) * teacher_before[name] + update_rate * student_params[name]
        assert torch.allclose(t_param, expected)
