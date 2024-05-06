from data.gen import Generator


class Database:
    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls.devices = Generator.get_devices()
            cls.jobs ,cls.tasks = Generator.get_jobs()
        return cls._instance

    @classmethod
    def get_all_devices(cls):
        return cls.devices
    # ---------- multiple ------------
    @classmethod
    def get_devices(cls, count):
        return cls.devices.head(count)

    @classmethod
    def get_jobs(cls, count):
        return cls.jobs.head(count)

    @classmethod
    def get_tasks(cls, count):
        return cls.tasks.head(count)
    # ---------- single ------------
    @classmethod
    def get_device(cls,id):
        return cls.devices.iloc[id]

    @classmethod
    def get_job(cls,id):
        return cls.jobs.iloc[id]

    @classmethod
    def get_task(cls, id):
        return cls.tasks.iloc[id]
    
    # ---------- helper  ------------
