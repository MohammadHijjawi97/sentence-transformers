from __future__ import annotations

from unittest.mock import MagicMock

import numpy as np
import pytest

from sentence_transformers.cross_encoder.evaluation import CrossEncoderCorrelationEvaluator


@pytest.mark.parametrize("show_progress_bar", [True, False])
def test_show_progress_bar_is_forwarded_to_predict(show_progress_bar: bool) -> None:
    model = MagicMock()
    model.predict.return_value = np.array([0.1, 0.5, 0.9])
    evaluator = CrossEncoderCorrelationEvaluator(
        sentence_pairs=[["a", "b"], ["c", "d"], ["e", "f"]],
        scores=[0.0, 0.5, 1.0],
        show_progress_bar=show_progress_bar,
    )

    evaluator(model)

    assert model.predict.call_args.kwargs["show_progress_bar"] is show_progress_bar
