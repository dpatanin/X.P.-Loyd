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
		private double upper;
		private double lower;
		private double upper2;
		private double lower2;
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
				N					= 24;
				X					= 18;
				Y					= 4;
				riskRewardRatio		= 1;
				multiplier			= 3; //MACD multiplier
				fast				= 12; //MACD Period fast
				slow				= 26; //MACD Period slow
				signal				= 9; //MACD Period signal
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
				AddChartIndicator(ATR(N));
			}
		}

		protected override void OnBarUpdate()
		{
			if(Time[0].DayOfWeek.ToString() == "Friday"
				&& Time[0].TimeOfDay.Hours == 23)
			{
				ExitLong();
				ExitShort();
				return;
			}
			else if(Time[0].DayOfWeek.ToString() == "Monday")
			{
				AfterStopLoss = false;
				startOfWeek = true;
			}
			else if(Time[0].DayOfWeek.ToString() == "Tuesday")
			{
				AfterStopLoss = true;
				startOfWeek = false;
			}
			double atr = X*ATR(N)[0]*0.25;
			if(Position.MarketPosition == MarketPosition.Long)//Long Position Exists
			{
				//StopLoss Long
				if(Close[0] < lower)
				{
					ExitLong();
				}
				//TakeProfit Long
				else if(Close[0] > upper)
				{
					lower = Close[0] - (atr / riskRewardRatio);
					upper = Close[0] + atr;
					PositionClosingPrice = Close[0];
				}
				else
				{
					lower = PositionClosingPrice - atr;
					upper = PositionClosingPrice + atr;
				}
			}
			else if(Position.MarketPosition == MarketPosition.Short)//Long Position Exists
			{
				//StopLoss Short
				if(Close[0] > upper)
				{
					ExitShort();
				}
				//TakeProfit Short
				else if(Close[0] < lower)
				{
					lower = Close[0] - atr;
					upper = Close[0] + (atr / riskRewardRatio);
					PositionClosingPrice = Close[0];
				}
				else
				{
					lower = PositionClosingPrice - atr;
					upper = PositionClosingPrice + atr;
				}
			}

			if(AfterStopLoss)
				UpdateAfterStopLoss();

			if(Position.MarketPosition == MarketPosition.Flat) //NoPositionExists
			{
				if(MACD(fast, slow, signal)[0] > MACD(fast, slow, signal).Avg[0]) //MACD line > Signal line
				{
					if(AfterStopLoss && Close[0] < upper2) //AfterStopLoss && (Current Price < PositionClosingPrice + y*ATR(N))
						return;
					else
					{
						EnterLong();
					}
				}
				else if(MACD(fast, slow, signal)[0] < MACD(fast, slow, signal).Avg[0]) //MACD line < Signal line
				{
					if(AfterStopLoss && Close[0] > lower2) //AfterStopLoss && (Current Price > PositionClosingPrice - y*ATR(N))
						return;
					else
					{
						EnterShort();
					}
				}
			}
			else
			{
				Values[0][0] = upper;
				Values[1][0] = lower;
			}
		}

		private void UpdateAfterStopLoss()
		{
			if(Close[0] < lower2)
			{
				AfterStopLoss = false;
			}
			else if(Close[0] > upper2)
			{
				AfterStopLoss = false;
			}
			Values[2][0] = lower2;
			Values[3][0] = upper2;
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
				double atr = Y*ATR(N)[0]*0.25;
				lower2 = AfterClosingPrice - atr;
				upper2 = AfterClosingPrice + atr;
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
		[Display(Name="N", Description="ATR Period", Order=1, GroupName="Parameters")]
		public int N
		{ get; set; }

		[NinjaScriptProperty]
		[Range(1, int.MaxValue)]
		[Display(Name="X", Order=2, GroupName="Parameters")]
		public int X
		{ get; set; }

		[NinjaScriptProperty]
		[Range(1, int.MaxValue)]
		[Display(Name="Y", Order=3, GroupName="Parameters")]
		public int Y
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
