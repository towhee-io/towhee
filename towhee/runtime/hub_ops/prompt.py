# Copyright 2023 Zilliz. All rights reserved.
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

from towhee.runtime.factory import HubOp

class Prompt:
    """
    QA prompt.
    """

    question_answer: HubOp = HubOp('prompt.question_answer')
    """
    __init__(self, temp: str = None, llm_name: str = None):
        temp(`str`):
            User-defined prompt, must contain {context} and {question}"
        llm_name(`str`):
            Pre-defined prompt, currently supports openai, ernie and dolly, openai prompt is used by default."

    __call__(self, question: str, context: str, history=Optional[List[Tuple]]) -> List[Dict[str, str]]:
        question(`str`):
            query string
        context(`str`):
            context string
        history(`List[Tuple]]`):
            history of chat, [(question1, answer1), (question2, answer2)]
        Return:
            A list of messages to set up chat. Must be a list of dictionaries with key value from "system", "question", "answer".
            For example, [{"question": "a past question?", "answer": "a past answer."}, {"question": "current question?"}]

    Example:

    .. code-block:: python

        from towhee import ops, pipe
        import requests

        towhee_docs = requests.get('https://raw.githubusercontent.com/towhee-io/towhee/main/README.md').content


        p = (
            pipe.input('question', 'docs', 'history')
            .map('docs', 'docs', lambda x: x[:2000])
            .map(('question', 'docs', 'history'), 'prompt', ops.prompt.question_answer())
            .map('prompt', 'answer', ops.LLM.OpenAI())
            .output('answer')
        )

        an1 = p('Tell me something about Towhee', towhee_docs, []).get()[0]
        print(an1)

        an2 = p('How to use it', towhee_docs, [('Tell me something about Towhee', an1)]).get()[0]
        print(an2)

    """

    template: HubOp = HubOp('prompt.template')
    """
    Prompt Template.

    __init__(self, temp: str, keys: List[str], sys_msg: str = None):
        temp(`str`):
            A template to create a prompt as the last user message.
        keys(`List[str]`):
            A list of keys used in template.
        sys_msg(`str`):
            A system message, defaults to None. If None, it will not pass any system message.

    __call__(self, *args) -> List[Dict[str, str]]:
        args:
            Depends on the template defined by the user
        Return:
            A list of messages to set up chat. Must be a list of dictionaries with key value from "system", "question", "answer".
            For example, [{"question": "a past question?", "answer": "a past answer."}, {"question": "current question?"}]

    Example:

    .. code-block:: python

        from towhee import ops, pipe
        import requests

        towhee_docs = requests.get('https://raw.githubusercontent.com/towhee-io/towhee/main/README.md').content

        temp = '''{question}

        input:
        {context}
        '''
        sys_message = 'Your name is TowheeChat.'

        p = (
            pipe.input('question', 'doc', 'history')
            .map('doc', 'doc', lambda x: x[:2000])
            .map(('question', 'doc', 'history'), 'prompt', ops.prompt.template(temp, ['question', 'context'], sys_message))
            .map('prompt', 'answer', ops.LLM.OpenAI())
            .output('answer')
        )

        an1 = p('What is your name?', [], []).get()[0]
        print(an1)

        an2 = p('Tell me something about Towhee', towhee_docs, []).get()[0]
        print(an2)

        an3 = p('How to use it', towhee_docs, [('Tell me something about Towhee', an2)]).get()[0]
        print(an3)

    """

    def __call__(self, *args, **kwargs):
        """
        Resolve the conflict issue that may be caused by ops users omitting the towhee namespace during use.
        """
        return HubOp('towhee.prompt')(*args, **kwargs)

