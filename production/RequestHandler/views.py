from django.shortcuts import render
from django.views.generic import TemplateView

from .exceptions import _handle_exceptions
from .functions import (
    random_forest,
    decision_tree,
    xgboost,
)


class Predict(TemplateView):
    template_name = "home.html"

    def get_context_data(self, **kwargs):
        context = super().get_context_data(**kwargs)
        context["models"] = [
            ["rf", "Random Forest Regression"],
            ["dt", "Decision Tree Regression"],
            ["xgb", "XGBoost"],
        ]
        return context

    def post(self, request, *args, **kwargs):
        try:
            data = request.POST.get("data")
            model = request.POST.get("model")
            _handle_exceptions(data)

            status, pred = switch[model](int(data))

            return render(
                request,
                "home.html",
                {
                    "input": data,
                    "pred": round(pred[0], 2),
                    "models": [
                        ["rf", "Random Forest Regression"],
                        ["dt", "Decision Tree Regression"],
                        ["xgb", "XGBoost"],
                    ],
                },
            )

        except Exception as error:
            return render(
                request,
                "home.html",
                {
                    "input": data,
                    "error": str(error),
                    "models": [
                        ["rf", "Random Forest Regression"],
                        ["dt", "Decision Tree Regression"],
                        ["xgb", "XGBoost"],
                    ],
                },
            )


switch = {
    "rf": random_forest,
    "dt": decision_tree,
    "xgb": xgboost,
}
