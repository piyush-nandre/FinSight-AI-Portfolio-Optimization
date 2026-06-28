def generate_recommendation(
    expected_return,
    volatility,
    sharpe,
    stress_impact,
    risk_profile
):

    score = 50

    # Return contribution
    if expected_return > 0.15:
        score += 15
    elif expected_return > 0.10:
        score += 10
    else:
        score += 5

    # Sharpe contribution
    if sharpe > 1.5:
        score += 15
    elif sharpe > 1:
        score += 10
    else:
        score += 5

    # Volatility penalty
    if volatility > 0.30:
        score -= 10
    elif volatility > 0.20:
        score -= 5

    # Stress test penalty
    if stress_impact < -0.10:
        score -= 15
    elif stress_impact < -0.05:
        score -= 10
    elif stress_impact < 0:
        score -= 5

    score = max(
        0,
        min(score, 100)
    )

    if score >= 80:
        recommendation = "STRONG BUY"

    elif score >= 65:
        recommendation = "BUY"

    elif score >= 50:
        recommendation = "HOLD"

    else:
        recommendation = "REDUCE"

    strengths = []
    risks = []

    if sharpe > 1:
        strengths.append(
            "Strong risk-adjusted returns"
        )

    if expected_return > 0.12:
        strengths.append(
            "Healthy expected return"
        )

    if volatility > 0.25:
        risks.append(
            "Portfolio volatility is elevated"
        )

    if stress_impact < -0.05:
        risks.append(
            "Sensitive to stress scenarios"
        )

    return {
        "score": score,
        "recommendation": recommendation,
        "strengths": strengths,
        "risks": risks
    }