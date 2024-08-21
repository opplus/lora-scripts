
class CeleryTaskRequest:
    def __init__(self, task_id:str, taskType: str, taskConfig: dict):
        self.task_id = task_id
        self.taskType = taskType
        self.taskConfig = taskConfig