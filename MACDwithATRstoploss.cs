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
				ExitPosition()
				return;
			}


			if (IsRecovering)
				OutOfRecoveryBoundsCheck();

			if (!IsRecovering)
			{
				bool pos = Position.MarketPosition
				double atr = activeBarrierScale * ATR(volatilityDampener)[0] * 0.25;
			
				if (pos == MarketPosition.Flat)
					EnterPosition();

				if (ShouldExitActive(pos))
				{
					HandleExit(atr);
					IsRecovering = true;
				}
				else
					UpdateATRBarrier(pos, atr);
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
		private bool ShouldExitActive(bool position)
		{
			return position == MarketPosition.Long && Close[0] < activeLower || position == MarketPosition.Short && Close[0] > activeUpper;
		}

		private void HandleExit(double atr)
		{
			ExitPosition();

			ExitClosingPrice = Close[0];

			//Sets red ATR-Barrier
			double atr = recoveryBarrierScale * ATR(volatilityDampener)[0] * 0.25;
			recoveryLower = ExitClosingPrice - atr;
			recoveryUpper = ExitClosingPrice + atr;
		}

		/// <summary>
		/// Updates the ATR Barrier and draws it on the chart using dots.
		/// Slides in profitable direction.
		/// </summary>
		private void UpdateATRBarrier(bool pos, double atr)
		{
			if (pos == MarketPosition.Long && Close[0] > activeUpper)
			{
				activeLower = Close[0] - (atr / riskRewardRatio);
				activeUpper = Close[0] + atr;
				EntryClosingPrice = Close[0];
			}
			else if (pos == MarketPosition.Short && Close[0] < activeLower)
			{
				activeLower = Close[0] - atr;
				activeUpper = Close[0] + (atr / riskRewardRatio);
				EntryClosingPrice = Close[0];
			}
			else
			{
				activeUpper = EntryClosingPrice + atr;
				activeLower = EntryClosingPrice - atr;
			}

			//draw
			Values[0][0] = activeUpper;
			Values[1][0] = activeLower;
		}

		/// <summary>
		/// Checks whether the price has moved beyond the specified limits after a StopLoss has been triggered
		/// </summary>
		private void OutOfRecoveryBoundsCheck()
		{
			IsRecovering = Close[0] > recoveryLower && Close[0] < recoveryUpper;

			//draw
			Values[2][0] = recoveryLower;
			Values[3][0] = recoveryUpper;
		}

		/// <summary>
		/// Enter Long or Short depending on the relationship between the MACD line and the Signal line
		/// </summary>
		private void EnterPosition()
		{
			bool MACDAboveSignal = MACD(fast, slow, signal)[0] > MACD(fast, slow, signal).Avg[0]
			if (MACDAboveSignal && Close[0] > recoveryUpper)
			{
				EnterLong();
				EntryClosingPrice = Close[0];
			}
			else if (!MACDAboveSignal && Close[0] < recoveryLower)
			{
				EnterShort();
				EntryClosingPrice = Close[0];
			}
		}

		private void ExitPosition()
		{
			ExitLong();
			ExitShort();
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
}
