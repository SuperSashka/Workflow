from env.estimator import estimate_runtime, estimate_transfer_time


class Context:
    def __init__(self, wf, cluster):
        self.wf = wf
        self.cluster = cluster
        self.schedule = self.init_schedule()

    def init_schedule(self):
        sched = dict()
        for n in self.cluster.nodes:
            sched[n.id] = []
        return sched

    def make_action(self, task_id, node_id):
        placed_item = self.find_time_slot(task_id, node_id)
        self.schedule[placed_item.node].append(placed_item)

    def is_ready_task(self, task_id):
        task = self.wf.id_to_task[task_id]
        for p in task.parents:
            if not p.is_head:
                if self.find_task_item(p.id) is None:
                    return False
        return True

    def is_not_scheduled(self, task_id):
        return self.find_task_item(task_id) is None

    def is_valid_action(self, task_id):
        return self.is_ready_task(task_id) and self.is_not_scheduled(task_id)


    def get_sorted_tasks(self):
        sorted_tasks = sorted([t.id for t in self.wf.tasks])
        return sorted_tasks

    def get_sorted_nodes(self):
        sorted_nodes = sorted([n.id for n in self.cluster.nodes])
        return sorted_nodes

    def find_time_slot(self, task_id, node_id):
        if not self.is_valid_action(task_id):
            raise Exception("Not valid action")
        task = self.wf.id_to_task[task_id]
        node = self.cluster.id_to_node[node_id]
        run_time = estimate_runtime(task, node)

        node_last_time = 0.0
        node_sched = self.schedule[node_id]
        if len(node_sched) > 0:
            node_last_time = node_sched[-1].end_time

        parent_times = [0.0]
        for parent in task.parents:
            if not parent.is_head:
                parent_item = self.find_task_item(parent.id)
                transfer_time = estimate_transfer_time(node, self.cluster.id_to_node[parent_item.node], task, self.wf.id_to_task[parent_item.task])
                parent_time = parent_item.end_time + transfer_time
                parent_times.append(parent_time)

        min_start_time = max(parent_times)
        start_time = max(min_start_time, node_last_time)
        end_time = start_time + run_time
        return ScheduleItem(task_id, node_id, start_time, end_time)

    def find_task_item(self, task_id):
        for n in self.schedule.keys():
            for it in self.schedule[n]:
                if it.task == task_id:
                    return it
        return None


class ScheduleItem:
    def __init__(self, task, node, st_time, end_time):
        self.task = task
        self.node = node
        self.st_time = st_time
        self.end_time = end_time
