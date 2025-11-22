"""
Verification Script for Passive Portfolio Backtest Results
Quick validation of key metrics and performance targets
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import pandas as pd
import numpy as np


def verify_results():
    """Verify backtest results meet expected criteria"""

    print("\n" + "=" * 70)
    print("          BACKTEST RESULTS VERIFICATION")
    print("=" * 70)

    try:
        # Load results CSV
        results_df = pd.read_csv('/Users/waiyang/Desktop/repo/dreamers-v2/src/umtrading/backtest_results.csv')

        # Calculate key metrics
        initial_value = results_df['Passive_Portfolio'].iloc[0]
        final_value = results_df['Passive_Portfolio'].iloc[-1]
        passive_return = (final_value - initial_value) / initial_value

        spy_initial = results_df['SPY_Benchmark'].iloc[0]
        spy_final = results_df['SPY_Benchmark'].iloc[-1]
        spy_return = (spy_final - spy_initial) / spy_initial

        alpha = passive_return - spy_return

        # Calculate Sharpe ratio
        passive_daily_returns = results_df['Passive_Daily_Return'].dropna()
        if len(passive_daily_returns) > 0:
            annual_return = (1 + passive_return) ** (252 / len(passive_daily_returns)) - 1
            volatility = passive_daily_returns.std() * np.sqrt(252)
            sharpe = (annual_return - 0.02) / volatility if volatility > 0 else 0
        else:
            sharpe = 0
            volatility = 0

        # Calculate max drawdown
        cumulative = (1 + passive_daily_returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_dd = drawdown.min()

        print("\n📊 KEY METRICS SUMMARY:")
        print("-" * 50)
        print(f"  Passive Portfolio Return:     {passive_return*100:.2f}%")
        print(f"  SPY Benchmark Return:         {spy_return*100:.2f}%")
        print(f"  Alpha (Excess Return):        {alpha*100:.2f}%")
        print(f"  Sharpe Ratio:                 {sharpe:.2f}")
        print(f"  Max Drawdown:                 {max_dd*100:.2f}%")
        print(f"  Volatility:                   {volatility*100:.2f}%")

        print("\n✅ VALIDATION CHECKS:")
        print("-" * 50)

        checks = [
            ("Total Return > 200%", passive_return > 2.0, passive_return*100),
            ("Alpha > 100%", alpha > 1.0, alpha*100),
            ("Sharpe Ratio > 1.0", sharpe > 1.0, sharpe),
            ("Max Drawdown < 40%", abs(max_dd) < 0.40, abs(max_dd)*100),
            ("Outperformance vs SPY", passive_return > spy_return, (passive_return - spy_return)*100),
        ]

        all_passed = True
        for check_name, passed, value in checks:
            status = "✅ PASS" if passed else "❌ FAIL"
            all_passed = all_passed and passed
            print(f"  {check_name:<30} {status}  (Value: {value:.2f})")

        print("\n" + "=" * 70)
        if all_passed:
            print("  🎉 ALL VALIDATION CHECKS PASSED! 🎉")
        else:
            print("  ⚠️  Some validation checks did not pass")
        print("=" * 70)

        # Performance tier assessment
        print("\n🏆 PERFORMANCE TIER ASSESSMENT:")
        print("-" * 50)

        if passive_return > 3.0 and sharpe > 1.2:
            tier = "EXCEPTIONAL"
            emoji = "🌟"
        elif passive_return > 2.5 and sharpe > 1.0:
            tier = "EXCELLENT"
            emoji = "⭐"
        elif passive_return > 2.0 and sharpe > 0.8:
            tier = "GOOD"
            emoji = "✨"
        elif passive_return > 1.5:
            tier = "SATISFACTORY"
            emoji = "👍"
        else:
            tier = "NEEDS IMPROVEMENT"
            emoji = "📈"

        print(f"  Performance Tier: {emoji} {tier}")
        print(f"  Strategy effectively captured {alpha/spy_return*100:.1f}% more return than the benchmark")

        # Risk-adjusted performance
        print("\n📈 RISK-ADJUSTED PERFORMANCE:")
        print("-" * 50)

        return_per_unit_risk = passive_return / abs(max_dd) if max_dd != 0 else 0
        print(f"  Return per unit of max drawdown: {return_per_unit_risk:.2f}")

        if sharpe > 0:
            information_ratio = alpha / (volatility * np.sqrt(252/len(passive_daily_returns)))
            print(f"  Information Ratio: {information_ratio:.2f}")

        # Trading efficiency
        print("\n💼 TRADING EFFICIENCY:")
        print("-" * 50)

        # Estimate from main backtest output
        total_trades = 63  # From backtest output
        total_costs = 1431.39  # From backtest output
        final_profit = (final_value - initial_value)

        cost_as_pct_of_profit = (total_costs / final_profit * 100) if final_profit > 0 else 0
        avg_rebalance_per_year = total_trades / 5  # 5 years

        print(f"  Total rebalances: {total_trades}")
        print(f"  Avg rebalances per year: {avg_rebalance_per_year:.1f}")
        print(f"  Trading costs as % of profit: {cost_as_pct_of_profit:.2f}%")

        if cost_as_pct_of_profit < 1:
            print(f"  Cost efficiency: EXCELLENT (< 1% of profit)")
        elif cost_as_pct_of_profit < 2:
            print(f"  Cost efficiency: GOOD (< 2% of profit)")
        else:
            print(f"  Cost efficiency: ACCEPTABLE")

        print("\n" + "=" * 70)
        print("  VERIFICATION COMPLETE")
        print("=" * 70)

        return {
            'passive_return': passive_return,
            'spy_return': spy_return,
            'alpha': alpha,
            'sharpe': sharpe,
            'max_dd': max_dd,
            'all_checks_passed': all_passed
        }

    except FileNotFoundError:
        print("\n❌ ERROR: backtest_results.csv not found!")
        print("   Please run the main backtest first: python3 run_backtest.py")
        return None
    except Exception as e:
        print(f"\n❌ ERROR during verification: {e}")
        return None


if __name__ == "__main__":
    results = verify_results()

    if results and results['all_checks_passed']:
        print("\n✅ Strategy is ready for production deployment!")
    elif results:
        print("\n⚠️ Strategy needs optimization before production deployment")
    else:
        print("\n❌ Verification failed - check backtest results")