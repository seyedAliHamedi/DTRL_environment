from data.gen import Generator
from sklearn.preprocessing import MinMaxScaler
import ast
class Database:

    def __init__(self):
        self._devices = Generator.get_devices()
        self._jobs, self._tasks = Generator.get_jobs()
        tasks_normalize = self._tasks.copy()
        for column in tasks_normalize.columns.values:
            if column in ("computational_load","input_size","output_size","is_safe","task_kind"):
                tasks_normalize[column] = (tasks_normalize[column] - tasks_normalize[column].min()) / (tasks_normalize[column].max() - tasks_normalize[column].min())
        self._task_norm=tasks_normalize
        
    # ------------ all ----------

    def get_all_devices(self):
        return self._devices.to_dict(orient='records')

    def get_all_jobs(self):
        return self._jobs.to_dict(orient='records')

    def get_all_tasks(self):
        return self._tasks.to_dict(orient='records')


    # ---------- single ------------

    def get_device(self, id):
        return self._devices.iloc[id].to_dict()

    def get_job(self, id):
        return self._jobs.iloc[id].to_dict()

    def get_task(self, id):
        return self._tasks.iloc[id].to_dict()
    
    def get_task_norm(self, id):
        return self._task_norm.iloc[id].to_dict()
    def task_pred_dec(self, id):
        self._tasks.at[id, "pred_count"] -= 1

    # ---------- helper  ------------

    def get_jobs_window(self, head, count):
        return self._jobs.iloc[head:head + count].to_dict(orient='records')
