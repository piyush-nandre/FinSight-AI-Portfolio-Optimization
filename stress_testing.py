ASSET_CATEGORIES = {

    "RELIANCE": "ENERGY",

    "HDFCBANK": "BANKING",
    "ICICIBANK": "BANKING",
    "BANKBEES": "BANKING",

    "INFY": "IT",
    "TCS": "IT",

    "LT": "INFRA",

    "NIFTYBEES": "MARKET",

    "GOLDBEES": "GOLD"
}


SCENARIO_SHOCKS = {

    "RBI Rate Hike": {

        "BANKING": -0.06,
        "IT": -0.03,
        "ENERGY": -0.02,
        "INFRA": -0.04,
        "MARKET": -0.05,
        "GOLD": 0.04
    },

    "IT Slowdown": {

        "IT": -0.12,
        "BANKING": -0.02,
        "ENERGY": -0.01,
        "INFRA": -0.02,
        "MARKET": -0.04,
        "GOLD": 0.03
    },

    "Market Correction": {

        "BANKING": -0.12,
        "IT": -0.12,
        "ENERGY": -0.10,
        "INFRA": -0.10,
        "MARKET": -0.15,
        "GOLD": 0.08
    },

    "Bull Market": {

        "BANKING": 0.18,
        "IT": 0.20,
        "ENERGY": 0.15,
        "INFRA": 0.16,
        "MARKET": 0.14,
        "GOLD": 0.03
    },

    "Recession": {

        "BANKING": -0.18,
        "IT": -0.15,
        "ENERGY": -0.12,
        "INFRA": -0.20,
        "MARKET": -0.18,
        "GOLD": 0.12
    },

    "Oil Price Shock": {

        "BANKING": -0.05,
        "IT": -0.03,
        "ENERGY": 0.12,
        "INFRA": -0.08,
        "MARKET": -0.06,
        "GOLD": 0.05
    },

    "Banking Crisis": {

        "BANKING": -0.25,
        "IT": -0.05,
        "ENERGY": -0.05,
        "INFRA": -0.08,
        "MARKET": -0.12,
        "GOLD": 0.15
    },

    "AI Boom": {

        "BANKING": 0.05,
        "IT": 0.30,
        "ENERGY": 0.04,
        "INFRA": 0.03,
        "MARKET": 0.12,
        "GOLD": -0.02
    },

    "China Slowdown": {

        "BANKING": -0.05,
        "IT": -0.03,
        "ENERGY": -0.15,
        "INFRA": -0.08,
        "MARKET": -0.07,
        "GOLD": 0.04
    }
}


def run_stress_test(
    weights,
    scenario
):

    scenario_shocks = (
        SCENARIO_SHOCKS[
            scenario
        ]
    )

    total_impact = 0

    affected_assets = []

    for asset, weight in (
        weights.items()
    ):

        asset_name = (
            asset.replace(
                ".NS",
                ""
            )
        )

        category = (
            ASSET_CATEGORIES.get(
                asset_name
            )
        )

        shock = (
            scenario_shocks.get(
                category,
                0
            )
        )

        impact = (
            weight * shock
        )

        total_impact += impact

        if abs(
            impact
        ) > 0.0001:

            affected_assets.append(
                (
                    asset,
                    impact
                )
            )

    affected_assets = sorted(
        affected_assets,
        key=lambda x: x[1]
    )

    negative_assets = [
        x for x in affected_assets
        if x[1] < 0
    ]

    positive_assets = [
        x for x in affected_assets
        if x[1] > 0
    ]

    top_losers = sorted(
        negative_assets,
        key=lambda x: x[1]
    )[:3]

    top_beneficiaries = sorted(
        positive_assets,
        key=lambda x: x[1],
        reverse=True
    )[:3]


    return {

        "impact":
            total_impact,

        "affected_assets":
            affected_assets,

        "top_losers":
            top_losers,

        "top_beneficiaries":
            top_beneficiaries
    }