import enum


class COVIDStatus(enum.Enum):
    PRE_COVID = 0
    POST_COVID = 1

    def __str__(self):
        return f"{self.name}"


ema_dictionary = {
    "Y1": "pam",
    "Y2": "phq4_score",
    "Y3": "phq2_score",
    "Y4": "gad2_score",
    "Y5": "social_level",
    "Y6": "sse_score",
    "Y7": "stress",
}
reverse_ema_dictionary = {v: k for k, v in ema_dictionary.items()}

physical_dictionary = {
    "P1": "excercise (seconds)",
    "P2": "studying (hours)",
    "P3": "in house (hours)",
    "P4": "sports (hours)",
}
social_dictionary = {
    "S1": "traveling (seconds)",
    "S2": "distance traveled (meters)",
    "S3": "time in social location (hours)",
    "S4": "visits",
    "S5": "duration unlocked phone in social locations (minutes)",
    "S6": "frequency of unlocked phone in social locations",
    "S7": "motion at social locations (minutes)",
}

sleep_dictionary = {
    "Z1": "sleep_duration",
    "Z2": "sleep start time",
    "Z3": "sleep end time",
}


demographic_dictionary = {
    "D1": "gender",
    "D2": "race",
    "D3": "os",
    "D4": "cohort year",
}


full_dictionary = (
    physical_dictionary
    | social_dictionary
    | sleep_dictionary
    | ema_dictionary
    | {"C": COVIDStatus}
    | demographic_dictionary
)

ema = [f"Y{i}" for i in range(1, 8, 1)]
physical = [f"P{i}" for i in range(1, 5, 1)]
social = [f"S{i}" for i in range(1, 8, 1)]
sleep = [f"Z{i}" for i in range(1, 4, 1)]
demographic = [f"D{i}" for i in range(1, 5, 1)]
