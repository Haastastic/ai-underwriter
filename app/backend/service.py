"""Pipeline orchestration for the API.

Holds the loaded model, its feature list, the training-derived cleaning
stats, a SHAP explainer, and an LLM client, and exposes one method per
pipeline step plus a combined `review`. Every method takes and returns plain
dicts, so the HTTP layer is a thin translation shell and this class can be
exercised directly in tests.
"""

from __future__ import annotations

from typing import Any

from src.data.clean import fit_clean_stats
from src.data.load import load_raw_data
from src.explain import build_explainer, explain_row
from src.features.prepare import prepare_application
from src.llm.adverse_action import generate_adverse_action
from src.llm.client import LLMClient
from src.model.artifacts import load_model
from src.model.decision import DENIED, decide

from app.backend.config import Settings
from app.backend.store import ReviewStore


class UnderwritingService:
    def __init__(
        self,
        settings: Settings,
        llm_client: LLMClient,
        llm_provider: str,
        store: ReviewStore,
    ):
        self.settings = settings
        self.llm_client = llm_client
        self.llm_provider = llm_provider
        self.store = store

        self.model, self.feature_names = load_model(settings.model_dir)
        self.clean_stats = fit_clean_stats(
            load_raw_data(settings.training_data_path)
        )
        self._explainer = build_explainer(self.model)

    # --- individual steps -------------------------------------------------

    def predict(self, application: dict[str, Any]) -> dict[str, Any]:
        row = prepare_application(
            application, self.clean_stats, self.feature_names
        )
        probability = float(self.model.predict_proba(row)[0, 1])
        return decide(
            probability,
            approve_below=self.settings.approve_below,
            deny_at_or_above=self.settings.deny_at_or_above,
        )

    def explain(self, application: dict[str, Any]) -> dict[str, Any]:
        row = prepare_application(
            application, self.clean_stats, self.feature_names
        )
        return explain_row(
            self.model,
            row.iloc[0].to_dict(),
            feature_names=self.feature_names,
            explainer=self._explainer,
        )

    def adverse_action(self, application: dict[str, Any]) -> dict[str, Any]:
        """Full statement of specific reasons; raises ValueError if not a denial."""
        decision = self.predict(application)
        explanation = self.explain(application)
        result = generate_adverse_action(
            explanation=explanation,
            decision=decision["decision"],
            max_reasons=self.settings.max_reasons,
            client=self.llm_client,
            model=self.settings.llm_model,
        )
        return {**result.to_dict(), "llm_provider": self.llm_provider}

    # --- combined + persisted ------------------------------------------

    def review(self, application: dict[str, Any], persist: bool = True) -> dict[str, Any]:
        decision = self.predict(application)
        explanation = self.explain(application)

        adverse_action: dict[str, Any] | None = None
        if decision["decision"] == DENIED:
            result = generate_adverse_action(
                explanation=explanation,
                decision=DENIED,
                max_reasons=self.settings.max_reasons,
                client=self.llm_client,
                model=self.settings.llm_model,
            )
            adverse_action = {**result.to_dict(), "llm_provider": self.llm_provider}

        record_id = None
        if persist:
            record_id = self.store.save(
                model_version=self.settings.model_version,
                probability=decision["probability"],
                decision=decision["decision"],
                application=application,
                explanation=explanation,
                adverse_action=adverse_action,
            )

        return {
            "id": record_id,
            "model_version": self.settings.model_version,
            "decision": decision,
            "explanation": explanation,
            "adverse_action": adverse_action,
        }
