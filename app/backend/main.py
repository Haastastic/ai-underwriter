"""FastAPI application wrapping the underwriting pipeline end to end.

    uvicorn app.backend.main:app --reload

Endpoints
    GET  /health
    POST /predict          application -> probability + three-band decision
    POST /explain          application -> structured SHAP contributions
    POST /adverse-action   application -> statement of specific reasons (denials only)
    POST /review           full pipeline, persisted; returns a record id
    GET  /applications     list stored review records (newest first)
    GET  /applications/{id}

`create_app()` builds the service from environment settings. Tests call it
with an injected service so the suite runs without a model artifact on disk
or a network call.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, FastAPI, HTTPException, Query, Request

from app.backend.config import Settings, settings_from_env
from app.backend.llm_client import build_llm_client
from app.backend.schemas import (
    AdverseActionOut,
    ApplicationIn,
    DecisionOut,
    ExplanationOut,
    HealthOut,
    ReviewOut,
    ReviewRecordOut,
)
from app.backend.service import UnderwritingService
from app.backend.store import ReviewStore


def build_service(settings: Settings | None = None) -> UnderwritingService:
    settings = settings or settings_from_env()
    llm_client, provider = build_llm_client()
    store = ReviewStore(settings.db_path)
    return UnderwritingService(settings, llm_client, provider, store)


def get_service(request: Request) -> UnderwritingService:
    return request.app.state.service


ServiceDep = Annotated[UnderwritingService, Depends(get_service)]


def create_app(
    settings: Settings | None = None,
    service: UnderwritingService | None = None,
) -> FastAPI:
    app = FastAPI(title="AI Underwriter", version="1.0.0")
    app.state.service = service or build_service(settings)

    @app.get("/health", response_model=HealthOut)
    def health(svc: ServiceDep) -> HealthOut:
        return HealthOut(
            status="ok",
            model_version=svc.settings.model_version,
            llm_provider=svc.llm_provider,
        )

    @app.post("/predict", response_model=DecisionOut)
    def predict(application: ApplicationIn, svc: ServiceDep) -> DecisionOut:
        return DecisionOut(**_run(svc.predict, application))

    @app.post("/explain", response_model=ExplanationOut)
    def explain(application: ApplicationIn, svc: ServiceDep) -> ExplanationOut:
        return ExplanationOut(**_run(svc.explain, application))

    @app.post("/adverse-action", response_model=AdverseActionOut)
    def adverse_action(application: ApplicationIn, svc: ServiceDep) -> AdverseActionOut:
        try:
            return AdverseActionOut(**_run(svc.adverse_action, application))
        except ValueError as exc:
            # decision was not a denial, or no reportable reasons
            raise HTTPException(status_code=409, detail=str(exc)) from exc

    @app.post("/review", response_model=ReviewOut)
    def review(application: ApplicationIn, svc: ServiceDep) -> ReviewOut:
        return ReviewOut(**_run(svc.review, application))

    @app.get("/applications", response_model=list[ReviewRecordOut])
    def list_applications(
        svc: ServiceDep,
        limit: Annotated[int, Query(ge=1, le=200)] = 50,
        offset: Annotated[int, Query(ge=0)] = 0,
        decision: str | None = None,
    ) -> list[ReviewRecordOut]:
        return [
            _record_out(r)
            for r in svc.store.list(limit=limit, offset=offset, decision=decision)
        ]

    @app.get("/applications/{review_id}", response_model=ReviewRecordOut)
    def get_application(review_id: int, svc: ServiceDep) -> ReviewRecordOut:
        record = svc.store.get(review_id)
        if record is None:
            raise HTTPException(status_code=404, detail="review not found")
        return _record_out(record)

    return app


def _run(step, application: ApplicationIn):
    try:
        return step(application.to_raw_dict())
    except ValueError:
        raise
    except Exception as exc:  # pragma: no cover - defensive
        raise HTTPException(status_code=500, detail=f"pipeline error: {exc}") from exc


def __getattr__(name: str):
    """Lazily build the ASGI app for `uvicorn app.backend.main:app`.

    Deferring construction keeps `from app.backend.main import create_app`
    cheap in tests, which build their own service and never need the
    environment-configured one (or a model artifact on disk).
    """
    if name == "app":
        app = create_app()
        globals()["app"] = app
        return app
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def _record_out(record: dict) -> ReviewRecordOut:
    return ReviewRecordOut(
        id=record["id"],
        created_at=record["created_at"],
        model_version=record["model_version"],
        application=record["application"],
        decision=DecisionOut(
            decision=record["decision"],
            probability=record["probability"],
            thresholds={},
        ),
        explanation=ExplanationOut(**record["explanation"]),
        adverse_action=(
            AdverseActionOut(**record["adverse_action"])
            if record["adverse_action"]
            else None
        ),
    )


