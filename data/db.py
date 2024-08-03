from data.gen import Generator


class Database:

    def __init__(self):
        self._devices = Generator.get_devices()
        self._jobs, self._tasks = Generator.get_jobs()

    # ------------ all ----------

    def get_all_devices(self):
        return self._devices.to_dict(orient='records')

    def get_all_jobs(self):
        return self._jobs.to_dict(orient='records')

    def get_all_tasks(self):
        return self._tasks.to_dict(orient='records')

    # ---------- multiple ------------

    def get_devices(self, count):
        return self._devices.head(count).to_dict(orient='records')

    def get_jobs(self, count):
        return self._jobs.head(count).to_dict(orient='records')

    def get_tasks(self, count):
        return self._tasks.head(count).to_dict(orient='records')

    # ---------- single ------------

    def get_device(self, id):
        return self._devices.iloc[id].to_dict()

    def get_job(self, id):
        return self._jobs.iloc[id].to_dict()

    def get_task(self, id):
        return self._tasks.iloc[id].to_dict()

    def task_pred_dec(self, id, column, new_val):
        self._tasks.at[id, "pred_count"] -= 1

    # ---------- helper  ------------

    def get_jobs_window(self, head, count):
        return self._jobs.iloc[head:head + count].to_dict(orient='records')

    def get_task_successors(self, task_ID):
        return self.get_task(task_ID)['successors']
