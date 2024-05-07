from data.gen import Generator


class Database:
    _devices = None
    _jobs = None
    _tasks = None

    @classmethod
    def load(cls):
        if not Database._devices: 
            Database._devices = Generator.get_devices()
        if not Database._jobs or not Database._tasks:
            Database._jobs, Database._tasks = Generator.get_jobs()
    # ------------ all ----------
    @classmethod
    def get_all_devices(cls):
        return cls._devices.to_dict(orient='records')

    @classmethod
    def get_all_jobs(cls):
        return cls._jobs.to_dict(orient='records')

    @classmethod
    def get_all_tasks(cls):
        return cls._tasks.to_dict(orient='records')


    # ---------- multiple ------------
    @classmethod
    def get_devices(cls, count):
        return cls._devices.head(count).to_dict(orient='records')

    @classmethod
    def get_jobs(cls, count):
        return cls._jobs.head(count).to_dict(orient='records')

    @classmethod
    def get_tasks(cls, count):
        return cls._tasks.head(count).to_dict(orient='records')

    # ---------- single ------------
    @classmethod
    def get_device(cls, id):
        return cls._devices.iloc[id].to_dict(orient='records')

    @classmethod
    def get_job(cls, id):
        return cls._jobs.iloc[id].to_dict(orient='records')

    @classmethod
    def get_task(cls, id):
        return cls._tasks.iloc[id].to_dict(orient='records')

    # ---------- helper  ------------
    @classmethod
    def get_jobs_window(cls,head,count):
        return cls._jobs.iloc[head:head+count].to_dict(orient='records')
