import json
import os
from datetime import datetime, timedelta

from hack import const, forecaster, scraper
from hack.logger import log


def generate_days() -> None:
    today = datetime.now().strftime("%Y-%m-%d")

    # Python weekday(): Mon=0 ... Sun=6
    # days since Sunday (Sun=0)
    now = datetime.now()
    days_since_sunday = (now.weekday() + 1) % 7
    sunday = (now - timedelta(days=days_since_sunday)).date()

    names = [
        "sunday",
        "monday",
        "tuesday",
        "wednesday",
        "thursday",
        "friday",
        "saturday",
    ]
    dates = [sunday + timedelta(days=i) for i in range(7)]
    days_names = {name: d.strftime("%Y-%m-%d") for name, d in zip(names, dates)}

    curiosity_weather = scraper.scrape_curiosity_weather()

    future_min_temps = forecaster.predict_next(
        process_func=lambda: [
            map["min_temp"] for map in curiosity_weather if map["min_temp"] != "--"
        ],
        n_lags=2000,
        horizon=7,
    )
    future_max_temps = forecaster.predict_next(
        process_func=lambda: [
            map["max_temp"] for map in curiosity_weather if map["max_temp"] != "--"
        ],
        n_lags=2000,
        horizon=7,
    )
    future_pressures = forecaster.predict_next(
        process_func=lambda: [
            map["pressure"] for map in curiosity_weather if map["pressure"] != "--"
        ],
        n_lags=2000,
        horizon=7,
    )
    future_sunrise_hours = forecaster.predict_next(
        process_func=lambda: [
            map["sunrise"][:2] for map in curiosity_weather if map["sunrise"] != "--"
        ],
        n_lags=2000,
        horizon=7,
    )
    future_sunrise_days = forecaster.predict_next(
        process_func=lambda: [
            map["sunrise"][3:] for map in curiosity_weather if map["sunrise"] != "--"
        ],
        n_lags=3,
        horizon=7,
    )
    future_sunset_hours = forecaster.predict_next(
        process_func=lambda: [
            float(map["sunset"][:2])
            for map in curiosity_weather
            if map["sunset"] != "--"
        ],
        n_lags=3,
        horizon=7,
    )
    future_sunset_days = forecaster.predict_next(
        process_func=lambda: [
            float(map["sunset"][3:])
            for map in curiosity_weather
            if map["sunset"] != "--"
        ],
        n_lags=3,
        horizon=7,
    )
    uv_irradiance_labels = ["Low", "Moderate", "High", "Very_High"]
    future_uv_irradiance_indexes = forecaster.predict_next(
        process_func=lambda: [
            uv_irradiance_labels.index(map["local_uv_irradiance_index"])
            for map in curiosity_weather
            if map["local_uv_irradiance_index"] != "--"
        ],
        n_lags=3,
        horizon=7,
    )

    days: dict[str, dict] = {}

    for name, date in days_names.items():
        for sol in curiosity_weather:
            if sol["terrestrial_date"] == date:
                days[name] = sol | {"predicted": False}
                break
        else:
            days[name] = dict(
                terrestrial_date=date,
                min_temp=str(f"{future_min_temps[0]:.0f}"),
                max_temp=str(f"{future_max_temps[0]:.0f}"),
                pressure=str(f"{future_pressures[0]:.0f}"),
                sunrise=str(
                    f"{round(future_sunrise_hours[0]):02}:{round(future_sunrise_days[0]):02}"
                ),
                sunset=str(
                    f"{round(future_sunset_hours[0]):02}:{round(future_sunset_days[0]):02}"
                ),
                local_uv_irradiance_index=str(
                    f"{uv_irradiance_labels[round(future_uv_irradiance_indexes[0])]}"
                ),
                predicted=True,
            )

            future_min_temps = future_min_temps[1:]
            future_max_temps = future_max_temps[1:]
            future_pressures = future_pressures[1:]
            future_sunrise_hours = future_sunrise_hours[1:]
            future_sunrise_days = future_sunrise_days[1:]
            future_sunset_hours = future_sunset_hours[1:]
            future_sunset_days = future_sunset_days[1:]
            future_uv_irradiance_indexes = future_uv_irradiance_indexes[1:]

    json.dump(days, open(os.path.join(const.flaskapp_dir_path, "days.json"), "w"))
