import threading
from towhee.engine.new_towhee_engine.action import Action


class Engine:
    """Engine Idea
    """
    def __init__(self, dag):
        """Create a Engine instance for the specific DC chain.

        This Engine takes a DAG and runs it locally using actions. An action is an object that
        contains an iterator, operator, and towhee array. The iterator reads from a previous action or is a generator,
        the operator is the calculations being performed, and the towhee array is where the results are stored.

        Args:
            dag (DataCollection.dag): The Dag created from the dataCollection

        Raises:
            NotImplementedError: _description_
        """
        self._dag = dag
        self._execution_plan = self.create_execution_plan()
        self._execution_roots = self.find_execution_roots()
        self._execution_leaves = self.find_execution_leaves()
        self._actions = self.generate_actions()


    def create_execution_plan(self) -> dict:
        """This method creates the execution plan fgit or the engine.

        The execution plan creates the connections between all the steps in the execution.
        Here the outlines for each action (wrapped op) are created. The outline contains:

        1. which action the current action is reading from
        2. which towhee array index it is reading from in that previous action
        3. which iterator is being used on the previous action (batch, window, generator)
        4. the current operator and its args
        5. how many readers for the current actions towhee.array, corresponds to how many
        actions are reading the current action.

        Raises:
            NotImplementedError: _description_

        Returns:
            dict: Dict of action plan connections.
        """
        raise NotImplementedError

    def find_execution_roots(self) -> list:
        """Find the roots of the DAG.

        Execution roots are the roots of the DAG. These roots are data sources for the chain. When
        a DC chain is executed with engine, the roots will begin feeding data, and coroutine execution will
        then take over.

        Raises:
            NotImplementedError: _description_

        Returns:
            dict: List of root IDs.
        """
        raise NotImplementedError

    def find_execution_leaves(self) -> list:
        """Find the leaves of the DAG.

        Execution leaves are the roots leaves of the DAG. These leaves are data sinks for the chain. When
        a DC chain is executed with engine, the leaves will contain the result data, and users will need to consume these.

        Raises:
            NotImplementedError: _description_

        Returns:
            dict: List of root IDs.
        """
        raise NotImplementedError

    def generate_actions(self) -> dict:
        """Generate the actions for execution.

        Returns:
            dict: Dict of actions.
        """
        actions = {}
        for key, val in self._execution_plan.items():
            self._actions[key] = Action(val)
        return actions

    def run(self):
        for x in self._execution_roots:
            y = threading.thread(x.start())
            y.start()
        return {x: iter(self._actions[x]) for x in self._execution_leaves}
