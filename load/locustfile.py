from locust import HttpUser, constant, task


class ForecastUser(HttpUser):
    wait_time = constant(0)

    @task
    def forecast(self):
        with self.client.post(
            "/predict",
            json={"forecast_days": 7},
            headers={"Host": "btc-forecast.local"},
            catch_response=True,
        ) as response:
            if response.status_code == 200 and len(response.json()["predictions"]) != 7:
                response.failure("Expected seven predictions.")
