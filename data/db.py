from data.gen import Generator
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
class Database:

    def __init__(self):
        self._devices = Generator.get_devices()
        self._jobs, self._tasks = Generator.get_jobs()
       
        self._task_norm = self._tasks.copy()
        self._task_norm=self.normalize_tasks(self._task_norm)
        
        self._devices_norm = self._devices.copy()
        self._devices_norm = self.normalize_devices(self._devices_norm )
        
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
    
    def get_device_norm(self,id):
        return self._devices_norm.iloc[id].values
    
    def task_pred_dec(self, id):
        self._tasks.at[id, "pred_count"] -= 1

    # ---------- helper  ------------

    def get_jobs_window(self, head, count):
        return self._jobs.iloc[head:head + count].to_dict(orient='records')

    # -------- normalize -------
    def normalize_tasks(self,tasks_normalize):
        for column in tasks_normalize.columns.values:
            if column in ("computational_load","input_size","output_size","is_safe"):
                tasks_normalize[column] = (tasks_normalize[column] - tasks_normalize[column].min()) / (tasks_normalize[column].max() - tasks_normalize[column].min())
        kinds=[1,2,3,4]
        for kind in kinds:
            tasks_normalize[f'kind{kind}'] = tasks_normalize['task_kind'].isin([kind]).astype(int)
        tasks_normalize.drop(['task_kind'],axis=1)
        return tasks_normalize
        
    def get_pe_data(self, id):
        pe = self.get_device(id)
        battery_capacity = pe['battery_capacity']
        battery_isl = pe['ISL']
        battery = (1 - battery_isl) * battery_capacity

        num_cores = pe['num_cores']

        devicePower = 0
        for index, core in enumerate(pe["voltages_frequencies"]):
            corePower = 0
            for mod in core:
                freq, vol = mod
                corePower += freq / vol
            devicePower += corePower
        devicePower = devicePower / num_cores

        error_rate = pe['error_rate']

        return [ devicePower, battery]

    def normalize_devices(self, devices_normalize):
        # Create empty lists to store the devicePower and battery values for all devices
        devicePower_list = []
        battery_list = []

        # Iterate through each device and calculate devicePower and battery
        for idx, device in devices_normalize.iterrows():
            devicePower, battery = self.get_pe_data(idx)
            devicePower_list.append(devicePower)
            battery_list.append(battery)

        # Convert the lists into a DataFrame to easily apply normalization
        df_normalize = pd.DataFrame({
            'devicePower': devicePower_list,
            'battery': battery_list
        })

        # Normalize the 'devicePower' and 'battery' columns using MinMaxScaler
        scaler = MinMaxScaler()
        df_normalize[['devicePower', 'battery']] = scaler.fit_transform(df_normalize[['devicePower', 'battery']])


        return df_normalize
