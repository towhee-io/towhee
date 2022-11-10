# Copyright 2021 Zilliz. All rights reserved.
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


from .node import Node


class Concat(Node):
    """Concatenates all the pipelins

       Examples:
           p1 = towhee.pipe.input('url').map('url', 'image', ops.image_decode.cv2())
           p2 = p1.map('image', 'vec1', ops.embedding.timm(model='resnet50'))
           p3 = p1.map('image', 'vec2', ops.embedding.timm(model='resnet50'))
           p1.concat(p2, p3).map(('vec1', 'vec2'), 'vec', ops.numpy_util.merge()).output('vec')
    """

    def initialize(self) -> bool:
        for q in self._in_ques:
            q.max_size = 0

        q_nums = len(self._in_ques)
        all_cols = []
        self.cols_every_que = []
        while q_nums > 0:
            cols = []
            schema = self._in_ques[q_nums - 1].schema
            for col in schema:
                if col not in all_cols:
                    cols.append(col)
            self.cols_every_que.append(cols)
            all_cols.extend(cols)
            q_nums -= 1
        self.cols_every_que.reverse()
        return True

    def process_step(self) -> bool:
        all_data = {}
        for i, q in enumerate(self._in_ques):
            data = q.get_dict(self.cols_every_que[i])
            if data:
                all_data.update(data)
        if not all_data:
            self._set_finished()
            return True

        for out_que in self._output_ques:
            if not out_que.put_dict(all_data):
                self._set_stopped()
                return True
        return False
