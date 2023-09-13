#region Using declarations
using System;
using System.Collections.Generic;
using System.ComponentModel;
using System.ComponentModel.DataAnnotations;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Windows;
using System.Windows.Input;
using System.Windows.Media;
using System.Xml.Serialization;
using NinjaTrader.Cbi;
using NinjaTrader.Gui;
using NinjaTrader.Gui.Chart;
using NinjaTrader.Gui.SuperDom;
using NinjaTrader.Gui.Tools;
using NinjaTrader.Data;
using NinjaTrader.NinjaScript;
using NinjaTrader.Core.FloatingPoint;
using NinjaTrader.NinjaScript.Indicators;
using NinjaTrader.NinjaScript.DrawingTools;
#endregion

//This namespace holds Strategies in this folder and is required. Do not change it. 
namespace NinjaTrader.NinjaScript.Strategies
{
	public class MACDwithATRstoploss : Strategy
	{
		private double EntryClosingPrice;
		private double ExitClosingPrice;
		private bool IsRecovering;
		private bool startOfWeek;
		private double activeUpper;
		private double activeLower;
		private double recoveryUpper;
		private double recoveryLower;
		private double focusLimitUpper;
		private double focusLimitLower;
		private int fast;
		private int slow;
		private int signal;

		protected override void OnStateChange()
		{
			if (State == State.SetDefaults)
			{
				Description = @"MACD with a constant ATR Stop Loss barrier zone";
				Name = "MACDwithATRstoploss";
				Calculate = Calculate.OnBarClose;
				EntriesPerDirection = 1;
				EntryHandling = EntryHandling.AllEntries;
				IsExitOnSessionCloseStrategy = true;
				ExitOnSessionCloseSeconds = 30;
				IsFillLimitOnTouch = false;
				MaximumBarsLookBack = MaximumBarsLookBack.TwoHundredFiftySix;
				OrderFillResolution = OrderFillResolution.Standard;
				Slippage = 0;
				StartBehavior = StartBehavior.WaitUntilFlat;
				TimeInForce = TimeInForce.Gtc;
				TraceOrders = false;
				RealtimeErrorHandling = RealtimeErrorHandling.StopCancelClose;
				StopTargetHandling = StopTargetHandling.PerEntryExecution;
				BarsRequiredToTrade = 0;
				// Disable this property for performance gains in Strategy Analyzer optimizations
				// See the Help Guide for additional information
				IsInstantiatedOnEachOptimizationIteration = true;
				focusType = FocusType.Linear;
				focusLimit = 0.2;
				focusStrength = 0.1;
				volatilityDampener = 12;
				activeBarrierScale = 8;
				recoveryBarrierScale = 4;
				riskRewardRatio = 1;
				multiplier = 2; //MACD multiplier
				fast = 12; //MACD Period fast
				slow = 26; //MACD Period slow
				signal = 9; //MACD Period signal
				AddPlot(new Stroke(Brushes.Green, 2), PlotStyle.Dot, "ActiveLower");
				AddPlot(new Stroke(Brushes.Green, 2), PlotStyle.Dot, "ActiveUpper");
				AddPlot(new Stroke(Brushes.Red, 2), PlotStyle.Dot, "RecoverLower");
				AddPlot(new Stroke(Brushes.Red, 2), PlotStyle.Dot, "RecoverUpper");
			}
			else if (State == State.Configure)
			{
				fast *= multiplier;
				slow *= multiplier;
				signal *= multiplier;
				AddChartIndicator(MACD(fast, slow, signal));
				AddChartIndicator(ATR(volatilityDampener));
			}
		}

		protected override void OnBarUpdate()
		{
			if (IsWeekend())
			{
				ExitPosition();
				return;
			}


			if (IsRecovering)
				OutOfRecoveryBoundsCheck();

			if (!IsRecovering)
			{
				bool justEntered = false;
				MarketPosition pos = Position.MarketPosition;
				double atr = ATR(volatilityDampener)[0] * 0.25;

				if (pos == MarketPosition.Flat)
					justEntered = EnterPosition();
				else if (ShouldExitActive(pos))
				{
					HandleExit(atr);
					IsRecovering = true;
				}

				UpdateATRBarrier(pos, atr, justEntered);
			}
		}

		/// <summary>
		/// Check for weekends, specifically at 11:00 PM on Fridays.
		/// </summary>
		private bool IsWeekend()
		{
			return Time[0].DayOfWeek.ToString() == "Friday" && Time[0].TimeOfDay.Hours == 23;
		}

		/// <summary>
		/// Checks if StopLoss is reached and position should be exited.
		/// </summary>
		private bool ShouldExitActive(MarketPosition position)
		{
			return position == MarketPosition.Long && Close[0] < activeLower || position == MarketPosition.Short && Close[0] > activeUpper;
		}

		private void HandleExit(double atr)
		{
			ExitPosition();

			ExitClosingPrice = Close[0];

			//Sets red ATR-Barrier
			double recoveryAtr = recoveryBarrierScale * atr;
			recoveryLower = ExitClosingPrice - recoveryAtr;
			recoveryUpper = ExitClosingPrice + recoveryAtr;
		}

		/// <summary>
		/// Updates the ATR Barrier and draws it on the chart using dots.
		/// Slides in profitable direction & sets focus bounds.
		/// </summary>
		private void UpdateATRBarrier(MarketPosition pos, double atr, bool justEntered)
		{

			bool slideUp = pos == MarketPosition.Long && Close[0] > activeUpper;
			bool slideDown = pos == MarketPosition.Short && Close[0] < activeLower;

			if (slideUp || slideDown)
				EntryClosingPrice = Close[0];
			
			double activeAtr = activeBarrierScale * atr;
			activeUpper = EntryClosingPrice + (slideDown ? activeAtr / riskRewardRatio : activeAtr);
			activeLower = EntryClosingPrice - (slideUp ? activeAtr / riskRewardRatio : activeAtr);

			if (justEntered || slideUp || slideDown)
				SetFocusBounds();

			//draw
			Values[0][0] = activeUpper;
			Values[1][0] = activeLower;
		}

		/// <summary>
		/// Checks whether the price has moved beyond the specified limits after a StopLoss has been triggered.
		/// </summary>
		private void OutOfRecoveryBoundsCheck()
		{
			IsRecovering = Close[0] > recoveryLower && Close[0] < recoveryUpper;

			//draw
			Values[2][0] = recoveryLower;
			Values[3][0] = recoveryUpper;
		}

		/// <summary>
		/// Enter Long or Short depending on the relationship between the MACD line and the Signal line.
		/// Returns true if position was entered.
		/// </summary>
		private bool EnterPosition()
		{
			bool goLong = MACD(fast, slow, signal)[0] > MACD(fast, slow, signal).Avg[0];
			
			if ((goLong && Close[0] > recoveryUpper) || (!goLong && Close[0] < recoveryLower))
			{
				if (goLong)
					EnterLong();
				else
					EnterShort();
				
				EntryClosingPrice = Close[0];
				return true;
			}

			return false;
		}

		private void ExitPosition()
		{
			ExitLong();
			ExitShort();
		}

		/// <summary>
		/// Sets the focused bounds after which no more narrowing should happen.
		/// </summary>
		private void SetFocusBounds()
		{
			double activeWidth = activeUpper - activeLower;
			double focusedWidth = activeWidth * focusLimit;
			double diff = (activeWidth - focusedWidth) / 2;

			focusLimitUpper = activeUpper - diff;
			focusLimitLower = activeLower + diff;
		}


		private double CalculateFocusDecline(double x, double strength, FocusType type)
		{
			if (strength <= 0)
			{
				throw new ArgumentException("Strength must be greater than zero.");
			}

			switch (type)
			{
				case FocusType.Linear:
					return x / strength;
				case FocusType.Exponential:
					return x * Math.Pow(strength, -1);
				case FocusType.Logarithmic:
					if (x <= 0)
					{
						throw new ArgumentException("x must be greater than zero for logarithmic focus.");
					}
					return x / Math.Log(x * strength);
				default:
					throw new ArgumentException("Invalid focus type.");
			}
		}

		#region Properties
		[NinjaScriptProperty]
		[Range(1, int.MaxValue)]
		[Display(Name = "Volatility Dampener", Description = "ATR period; The greater, the less reactive to volatility.", Order = 1, GroupName = "Parameters")]
		public int volatilityDampener
		{ get; set; }

		[NinjaScriptProperty]
		[Range(1, int.MaxValue)]
		[Display(Name = "Active Barrier Scale", Description = "Green barriers", Order = 2, GroupName = "Parameters")]
		public int activeBarrierScale
		{ get; set; }

		[NinjaScriptProperty]
		[Range(1, int.MaxValue)]
		[Display(Name = "Recovery Barrier Scale", Description = "Red barriers", Order = 3, GroupName = "Parameters")]
		public int recoveryBarrierScale
		{ get; set; }

		[NinjaScriptProperty]
		[Range(1, int.MaxValue)]
		[Display(Name = "MACD Multiplier", Order = 4, GroupName = "Parameters")]
		public int multiplier
		{ get; set; }

		[NinjaScriptProperty]
		[Range(1, int.MaxValue)]
		[Display(Name = "Risk/Reward Ratio", Order = 5, GroupName = "Parameters")]
		public int riskRewardRatio
		{ get; set; }

		[NinjaScriptProperty]
		[Display(Name = "Focus Type", Order = 6, GroupName = "Parameters")]
		public FocusType focusType
		{ get; set; }

		[NinjaScriptProperty]
		[Display(Name = "Focus Limit", Order = 7, Description = "Percentage limit to which focus converges", GroupName = "Parameters")]
		public double focusLimit
		{ get; set; }

		[NinjaScriptProperty]
		[Display(Name = "Focus Strength", Order = 8, Description = "Rate at which focus converges; Liner: percentage decline; Else: 0-1 where 0 is no decline.", GroupName = "Parameters")]
		public double focusStrength
		{ get; set; }
		#endregion
	}

	public enum FocusType
	{
		None,
		Linear,
		Exponential,
		Logarithmic
	}
}
