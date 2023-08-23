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
		private double StopLossBarrierLong;
		private double StopLossBarrierShort;
		private double PositionClosingPrice;
		private double AfterClosingPrice;
		private bool AfterStopLoss;
		private double upper;
		private double lower;
		private double upper2;
		private double lower2;
		private double avgPrice;
		
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
				N					= 12;
				X					= 6;
				Y					= 2;
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
				AddChartIndicator(MACD(fast,slow,signal));
				AddChartIndicator(ATR(N));
			}
			else if (State == State.DataLoaded)
			{
				PositionClosingPrice = Close[0];
			}
		}

		protected override void OnBarUpdate()
		{
			if(Position.MarketPosition == MarketPosition.Long)//Long Position Exists
			{
				//StopLoss Long
				if(Close[0] < lower)
				{
					Print("Exit Long");
					ExitLong();
				}
				//TakeProfit Long
				else if(Close[0] > upper)
				{
					lower = Close[0] - X*ATR(N)[0]*0.25;
					upper = Close[0] + X*ATR(N)[0]*0.25;
				}
			}
			else if(Position.MarketPosition == MarketPosition.Short)//Long Position Exists
			{
				//StopLoss Short
				if(Close[0] > upper)
				{
					Print("Exit Long");
					ExitShort();
				}
				//TakeProfit Short
				else if(Close[0] < lower)
				{
					lower = Close[0] - X*ATR(N)[0]*0.25;
					upper = Close[0] + X*ATR(N)[0]*0.25;
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
						Print("Enter Long");
						EnterLong();
					}
				}
				else if(MACD(fast, slow, signal)[0] < MACD(fast, slow, signal).Avg[0]) //MACD line < Signal line
				{
					if(AfterStopLoss && Close[0] > lower2) //AfterStopLoss && (Current Price > PositionClosingPrice - y*ATR(N))
						return;
					else
					{
						Print("Enter Short");
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
				
				//Resets StopLoss and ProfitTaget
				//SetStopLoss(CalculationMode.Ticks, X*ATR(N)[0]);
				//SetProfitTarget(CalculationMode.Ticks, X*ATR(N)[0]);
				
				//Sets red ATR-Barrier
				lower2 = AfterClosingPrice - Y*ATR(N)[0]*0.25;
				upper2 = AfterClosingPrice + Y*ATR(N)[0]*0.25;
				
				Print("Exit last Position");
			}
			
			//Position Opened
			else
			{
				avgPrice = position.AveragePrice;
				
				//Set StopLoss and ProfitTarget
				//SetStopLoss(CalculationMode.Ticks, X*ATR(N)[0]);
				//SetProfitTarget(CalculationMode.Ticks, X*ATR(N)[0]);
				
				//Sets green ATR-Barrier
				lower = Close[0] - X*ATR(N)[0]*0.25;
				upper = Close[0] + X*ATR(N)[0]*0.25;
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
		[Display(Name="fast", Order=4, GroupName="Parameters")]
		public int fast
		{ get; set; }
		
		[NinjaScriptProperty]
		[Range(1, int.MaxValue)]
		[Display(Name="slow", Order=5, GroupName="Parameters")]
		public int slow
		{ get; set; }
		
		[NinjaScriptProperty]
		[Range(1, int.MaxValue)]
		[Display(Name="signal", Order=6, GroupName="Parameters")]
		public int signal
		{ get; set; }
		#endregion
	}
}
