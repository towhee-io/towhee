# Built on top of the original implementation at https://github.com/albanie/collaborative-experts
#
# Modifications by Copyright 2022 Zilliz. All rights reserved.
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

def expert_tensor_storage(experts, feat_aggregation):
    expert_storage = {"fixed": set(), "variable": set(), "flaky": set()}
    # fixed_sz_experts, variable_sz_experts, flaky_experts = set(), set(), set()
    for expert, config in feat_aggregation.items():
        if config["temporal"] in {"vlad"}:
            expert_storage["variable"].add(expert)
        elif all([x in {"avg", "max", "ent", "std"} for x in  # pylint: disable=use-a-generator
                  config["temporal"].split("-")]):  # pylint: disable=use-a-generator
            expert_storage["fixed"].add(expert)
        else:
            raise ValueError(f"unknown temporal strategy: {config['temporal']}")
        if config.get("flaky", False):
            expert_storage["flaky"].add(expert)

    for key, value in expert_storage.items():
        expert_storage[key] = value.intersection(set(experts))
    return expert_storage
