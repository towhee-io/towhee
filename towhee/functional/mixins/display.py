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

import numpy


def check_list(query, query_titles, data_ids, data_titles):
    assert len(query) == len(data_ids)
    if not query_titles:
        query_titles = [None for _ in range(len(query))]
    if not data_titles:
        data_titles = numpy.zeros((len(data_ids), len(data_ids[0]))).tolist()
    return query_titles, data_titles


class PlotMixin:
    """
    Mixin for plot image in notebook
    """
    def plot(self):
        from towhee.utils.plot_utils import plot_img # pylint: disable=C
        for img in self._iterable:
            plot_img(img)

    @classmethod
    def plot_results(cls, query: list, data: list, res_ids: list, query_titles: list = None, res_titles: list = None):
        """
        Plot the results of reverse image search.

        Args:
            query (`list`):
                Query image list, such as ['1.jpg', '2.jpg'].
            data (`list`):
                Dataset of image, such as ['11.jpg', '22.jpg', '33.jpg'].
            res_ids (`list`):
                The result ids of the reverse image search, used to find the loc of the result image in the dataset. If seacheed topk = 3 and the
                number of queries is 2, and res_ids = [[0, 2, 1], [1, 0, 2]].
            query_titles (`list`):
                Titles of the query list to show with matplotlib.
            res_titles (`list`):
                Titles of the result ids to show with matplotlib.
        """
        from towhee.utils.plot_utils import plot_img, plot_img_list # pylint: disable=C
        query_titles, res_titles = check_list(query, query_titles, res_ids, res_titles)

        for i, q_title in zip(range(len(query)), query_titles):
            plot_img(query[i], q_title)
            plot_img_list(data, res_ids[i], res_titles[i])
