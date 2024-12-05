# Copyright 2024 NVIDIA CORPORATION & AFFILIATES
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
#
# SPDX-License-Identifier: Apache-2.0

from collections.abc import Iterable
from itertools import repeat
from typing import Union, Tuple

import torch
from torch.utils.checkpoint import checkpoint, checkpoint_sequential


def _ntuple(n):
    def parse(x):
        if isinstance(x, Iterable) and not isinstance(x, str):
            return x
        return tuple(repeat(x, n))

    return parse


to_1tuple = _ntuple(1)
to_2tuple = _ntuple(2)


def auto_grad_checkpoint(module, *args, **kwargs):
    if getattr(module, "grad_checkpointing", False):
        if isinstance(module, Iterable):
            gc_step = module[0].grad_checkpointing_step
            return checkpoint_sequential(module, gc_step, *args, **kwargs)
        else:
            return checkpoint(module, *args, **kwargs)
    return module(*args, **kwargs)


def checkpoint_sequential(functions, step, input, *args, **kwargs):

    # Hack for keyword-only parameter in a python 2.7-compliant way
    preserve = kwargs.pop("preserve_rng_state", True)
    if kwargs:
        raise ValueError("Unexpected keyword arguments: " + ",".join(arg for arg in kwargs))

    def run_function(start, end, functions):
        def forward(input):
            for j in range(start, end + 1):
                input = functions[j](input, *args)
            return input

        return forward

    if isinstance(functions, torch.nn.Sequential):
        functions = list(functions.children())

    # the last chunk has to be non-volatile
    end = -1
    segment = len(functions) // step
    for start in range(0, step * (segment - 1), step):
        end = start + step - 1
        input = checkpoint(run_function(start, end, functions), input, preserve_rng_state=preserve)
    return run_function(end + 1, len(functions) - 1, functions)(input)

def val2list(x: list or tuple or any, repeat_time=1) -> list:  # type: ignore
    """Repeat `val` for `repeat_time` times and return the list or val if list/tuple."""
    if isinstance(x, (list, tuple)):
        return list(x)
    return [x for _ in range(repeat_time)]

def val2tuple(x: list or tuple or any, min_len: int = 1, idx_repeat: int = -1) -> tuple:  # type: ignore
    """Return tuple with min_len by repeating element at idx_repeat."""
    # convert to list first
    x = val2list(x)

    # repeat elements if necessary
    if len(x) > 0:
        x[idx_repeat:idx_repeat] = [x[idx_repeat] for _ in range(min_len - len(x))]

    return tuple(x)

def get_same_padding(kernel_size: Union[int, Tuple[int, ...]]) -> Union[int, Tuple[int, ...]]:
    if isinstance(kernel_size, tuple):
        return tuple([get_same_padding(ks) for ks in kernel_size])
    else:
        assert kernel_size % 2 > 0, f"kernel size {kernel_size} should be odd number"
        return kernel_size // 2
