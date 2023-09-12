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
		private double PositionClosingPrice;
		private double AfterClosingPrice;
		private bool AfterStopLoss;
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
				Description									= @"MACD with a constant ATR Stop Loss barrier zone";
				Name										= "MACDwithATRstoploss";
				Calculate									= Calculate.OnBarClose;
				EntriesPerDirection							= 1;
				EntryHandling								= EntryHandling.AllEntries;
				IsExitOnSessionCloseStrategy				= true;
				ExitOnSessionCloseSeconds					= 30;
				IsFillLimitOnTouch							= false;
				MaximumBarsLookBack							= MaximumBarsLookBack.TwoHundredFiftySix;
				OrderFillResolution							= OrderFillResolution.Standard;
				Slippage									= 0;
				StartBehavior								= StartBehavior.WaitUntilFlat;
				TimeInForce									= TimeInForce.Gtc;
				TraceOrders									= false;
				RealtimeErrorHandling						= RealtimeErrorHandling.StopCancelClose;
				StopTargetHandling							= StopTargetHandling.PerEntryExecution;
				BarsRequiredToTrade							= 0;
				// Disable this property for performance gains in Strategy Analyzer optimizations
				// See the Help Guide for additional information
				IsInstantiatedOnEachOptimizationIteration	= true;
				volatilityDampener		= 12;
				activeBarrierScale		= 8;
				recoveryBarrierScale	= 4;
				riskRewardRatio			= 1;
				multiplier				= 2; //MACD multiplier
				fast					= 12; //MACD Period fast
				slow					= 26; //MACD Period slow
				signal					= 9; //MACD Period signal
				AddPlot(new Stroke(Brushes.Green, 2), PlotStyle.Dot, "Lower");
				AddPlot(new Stroke(Brushes.Green, 2), PlotStyle.Dot, "Upper");
				AddPlot(new Stroke(Brushes.Red, 2), PlotStyle.Dot, "Lower2");
				AddPlot(new Stroke(Brushes.Red, 2), PlotStyle.Dot, "Upper2");
			}
			else if (State == State.Configure)
			{
				fast *= multiplier;
				slow *= multiplier;
				signal *= multiplier;
				AddChartIndicator(MACD(fast,slow,signal));
				AddChartIndicator(ATR(volatilityDampener));
			}
		}

		protected override void OnBarUpdate()
		{
			double atr = activeBarrierScale*ATR(volatilityDampener)[0]*0.25;
			SkipWeekends();

			if(!AfterStopLoss)
				UpdateATRBarrier(atr);

			OutOfBoundsCheck(atr);

			if(AfterStopLoss)
				OutOfBoundsCheckAfterStopLoss();

			Trade();
		}
		/// <summary>
		/// Closes the current trading position on weekends,
		/// specifically at 11:00 PM on Fridays, and then resumes trading once Mondays begin.
		/// </summary>

		private void SkipWeekends()
		{
			if(Time[0].DayOfWeek.ToString() == "Friday"
				&& Time[0].TimeOfDay.Hours == 23)
			{
				ExitLong();
				ExitShort();
				return;
			}
		}

		/// <summary>
		/// Checks if StopLoss/TakeProfit signals are reached.
		///
		/// Case 1: Sell current position because of StopLoss.
		/// Case 2: Slide ATR-Barrier in profitable direction.
		/// Case 3: Barrier is not reached, ATR-Barrier is being updated.
		/// </summary>
		private void OutOfBoundsCheck(double atr)
		{
			//Long Position Exist
			if(Position.MarketPosition == MarketPosition.Long)
			{
				//StopLoss Long
				if(Close[0] < activeLower)
				{
					ExitLong();
				}
				//Slide ATR-Barrier in profitable direction
				else if(Close[0] > activeUpper)
				{
					activeLower = Close[0] - (atr / riskRewardRatio);
					activeUpper = Close[0] + atr;
					PositionClosingPrice = Close[0];
				}
			}

			//Short Position Exists
			else if(Position.MarketPosition == MarketPosition.Short)
			{
				//StopLoss Short
				if(Close[0] > activeUpper)
				{
					ExitShort();
				}
				//Slide ATR-Barrier in profitable direction
				else if(Close[0] < activeLower)
				{
					activeLower = Close[0] - atr;
					activeUpper = Close[0] + (atr / riskRewardRatio);
					PositionClosingPrice = Close[0];
				}
			}
		}

		/// <summary>
		/// It updates the ATR Barrier and draws it on the chart using dots.
		/// </summary>
		private void UpdateATRBarrier(double atr)
		{
			//update
			activeUpper = PositionClosingPrice + atr;
			activeLower = PositionClosingPrice - atr;
			//draw
			Values[0][0] = activeUpper;
			Values[1][0] = activeLower;
		}

		/// <summary>
		/// This checks whether the price has moved beyond the specified limits after a StopLoss has been triggered
		/// </summary>
		private void OutOfBoundsCheckAfterStopLoss()
		{
			if(Close[0] < recoveryLower)
			{
				AfterStopLoss = false;
			}
			else if(Close[0] > recoveryUpper)
			{
				AfterStopLoss = false;
			}
			Values[2][0] = recoveryLower;
			Values[3][0] = recoveryUpper;
		}

		/// <summary>
		/// Checks if the price is outside the AfterStopLoss-Barrier.
		/// If it is, then go Long or Short depending on
		/// the relationship between the MACD line and the Signal line
		/// </summary>
		private void Trade()
		{
			//NoPositionExists
			if(Position.MarketPosition == MarketPosition.Flat)
			{
				//MACD line > Signal line
				if(MACD(fast, slow, signal)[0] > MACD(fast, slow, signal).Avg[0])
				{

					if(AfterStopLoss && Close[0] < recoveryUpper)
						return;
					else
					{
						EnterLong();
					}
				}
				//MACD line < Signal line
				else if(MACD(fast, slow, signal)[0] < MACD(fast, slow, signal).Avg[0])
				{
					// If inside the AfterStopLoss-Barrier, then return, else go Short.
					if(AfterStopLoss && Close[0] > recoveryLower)
						return;
					else
					{
						EnterShort();
					}
				}
			}
		}

		protected override void OnPositionUpdate(Cbi.Position position,
			double averagePrice, int quantity, Cbi.MarketPosition marketPosition)
		{
			//Position Closed
			if (position.MarketPosition == MarketPosition.Flat)
			{
				AfterStopLoss = true;
				AfterClosingPrice = Close[0];

				//Sets red ATR-Barrier
				double atr = recoveryBarrierScale*ATR(volatilityDampener)[0]*0.25;
				recoveryLower = AfterClosingPrice - atr;
				recoveryUpper = AfterClosingPrice + atr;
			}

			//Position Opened
			else
			{
				PositionClosingPrice = Close[0];
			}
		}

		#region Properties
		[NinjaScriptProperty]
		[Range(1, int.MaxValue)]
		[Display(Name="VolatilityDampener", Description="ATR period; The greater, the less reactive to volatility.", Order=1, GroupName="Parameters")]
		public int volatilityDampener
		{ get; set; }

		[NinjaScriptProperty]
		[Range(1, int.MaxValue)]
		[Display(Name="ActiveBarrierScale", Description="Green barriers", Order=2, GroupName="Parameters")]
		public int activeBarrierScale
		{ get; set; }

		[NinjaScriptProperty]
		[Range(1, int.MaxValue)]
		[Display(Name="RecoveryBarrierScale", Description="Red barriers", Order=3, GroupName="Parameters")]
		public int recoveryBarrierScale
		{ get; set; }

		[NinjaScriptProperty]
		[Range(1, int.MaxValue)]
		[Display(Name="MACD multiplier", Order=4, GroupName="Parameters")]
		public int multiplier
		{ get; set; }

		[NinjaScriptProperty]
		[Range(1, int.MaxValue)]
		[Display(Name="Risk/Reward Ratio", Order=5, GroupName="Parameters")]
		public int riskRewardRatio
		{ get; set; }
		#endregion
	}
}
