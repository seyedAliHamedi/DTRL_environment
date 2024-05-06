from configs import config

class Generator:



    def generateIot(self):
        devices_data_IOT = []
        for i in range(num_IOT_devices):
            cpu_cores = np.random.choice([4, 6, 8])
            device_info = {
                "id": i,
                "number_of_cpu_cores": cpu_cores,
                "occupied_cores": [np.random.choice([0, 1]) for _ in range(cpu_cores)],
                "voltages_frequencies": [
                    [
                        voltages_frequencies_IOT[i]
                        for i in np.random.choice(6, size=4, replace=False)
                    ]
                    for core in range(cpu_cores)
                ],
                "ISL": np.random.randint(10, 21),
                "capacitance": [np.random.uniform(2, 3) * 1e-9 for _ in range(cpu_cores)],
                "powerIdle": [
                    np.random.choice([700, 800, 900]) * 1e-6 for _ in range(cpu_cores)
                ],
                "batteryLevel": np.random.randint(36, 41) * 1e9,
                "errorRate": np.random.randint(1, 6) / 100,
                "accetableTasks": np.random.choice(
                    task_kinds, size=np.random.randint(2, 5), replace=False
                ),
                "handleSafeTask": np.random.choice([0, 1], p=[0.25, 0.75]),
            }
            devices_data_IOT.append(device_info)

        IoTdevices = pd.DataFrame(devices_data_IOT)

        IoTdevices.set_index("id", inplace=True)
        IoTdevices["name"] = "iot"
        IoTdevices

