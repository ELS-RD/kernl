#  Copyright 2022 Lefebvre Sarrut
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
#

from typing import Iterable

from torch.fx import Node


def compare_static_args(arg1, arg2):
    if type(arg1) != type(arg2):
        return False
    if isinstance(arg1, Node) and isinstance(arg2, Node):
        return True
    if isinstance(arg1, Iterable):
        if len(arg1) != len(arg2):
            return False
        return all([compare_static_args(arg1[j], arg2[j]) for j in range(0, len(arg1))])
    return arg1 == arg2


def static_args_are_equal(pn: Node, gn: Node) -> bool:
    if len(pn.args) != len(gn.args):
        return False
    for i in range(0, len(pn.args)):
        if not compare_static_args(pn.args[i], gn.args[i]):
            return False
    return True
