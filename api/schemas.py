"""
schemas.py — Pydantic request/response models for the FastAPI endpoint.
"""

from pydantic import BaseModel, Field
from typing import List


class Transaction(BaseModel):
    """A single credit card transaction (30 features)."""
    Time: float = Field(..., description="Seconds elapsed since first transaction")
    V1:   float = Field(0.0);  V2:  float = Field(0.0)
    V3:   float = Field(0.0);  V4:  float = Field(0.0)
    V5:   float = Field(0.0);  V6:  float = Field(0.0)
    V7:   float = Field(0.0);  V8:  float = Field(0.0)
    V9:   float = Field(0.0);  V10: float = Field(0.0)
    V11:  float = Field(0.0);  V12: float = Field(0.0)
    V13:  float = Field(0.0);  V14: float = Field(0.0)
    V15:  float = Field(0.0);  V16: float = Field(0.0)
    V17:  float = Field(0.0);  V18: float = Field(0.0)
    V19:  float = Field(0.0);  V20: float = Field(0.0)
    V21:  float = Field(0.0);  V22: float = Field(0.0)
    V23:  float = Field(0.0);  V24: float = Field(0.0)
    V25:  float = Field(0.0);  V26: float = Field(0.0)
    V27:  float = Field(0.0);  V28: float = Field(0.0)
    Amount: float = Field(..., description="Transaction amount in USD", ge=0)

    def to_feature_list(self) -> list:
        """Return features as [Time, V1..V28, Amount]."""
        return [
            self.Time,
            self.V1,  self.V2,  self.V3,  self.V4,  self.V5,
            self.V6,  self.V7,  self.V8,  self.V9,  self.V10,
            self.V11, self.V12, self.V13, self.V14, self.V15,
            self.V16, self.V17, self.V18, self.V19, self.V20,
            self.V21, self.V22, self.V23, self.V24, self.V25,
            self.V26, self.V27, self.V28,
            self.Amount
        ]

    model_config = {
        "json_schema_extra": {
            "examples": [{
                "Time": 406.0, "V1": -2.3122265423263, "V2": 1.95199201064158,
                "V3": -1.60985073229769, "V4": 3.9979055875468, "V5": -0.52238753956764,
                "V6": -1.42654531920595, "V7": -2.53738730624579, "V8": 1.39165724829804,
                "V9": -2.77008927719433, "V10": -2.77227214465915, "V11": 3.20203320709635,
                "V12": -2.89990738849473, "V13": -0.595221881324605, "V14": -4.28925378244217,
                "V15": 0.389724120274487, "V16": -1.14074717980657, "V17": -2.83005567450437,
                "V18": -0.0168224681808405, "V19": 0.416955705037907, "V20": 0.126910559061474,
                "V21": 0.517232370861764, "V22": -0.0350493686052974, "V23": -0.465211076182388,
                "V24": 0.320198197514179, "V25": 0.0445191674731724, "V26": 0.177839798284401,
                "V27": 0.261145002567658, "V28": -0.143275874698919, "Amount": 149.62
            }]
        }
    }


class PredictionResponse(BaseModel):
    prediction:  str   # "FRAUD" or "LEGITIMATE"
    confidence:  float # 0.0 – 1.0
    risk_level:  str   # "LOW" | "MEDIUM" | "HIGH"


class BatchRequest(BaseModel):
    transactions: List[Transaction]


class BatchResponse(BaseModel):
    results: List[PredictionResponse]
    total:   int
    fraud_count: int


class HealthResponse(BaseModel):
    status:  str
    model:   str
    version: str


class StatsResponse(BaseModel):
    total_transactions: int
    fraud_transactions: int
    fraud_rate:         float
    best_model:         str
    best_model_f1:      float
    best_model_recall:  float
