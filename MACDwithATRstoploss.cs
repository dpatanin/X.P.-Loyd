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
				BarsRequiredToTrade							= 20;
				// Disable this property for performance gains in Strategy Analyzer optimizations
				// See the Help Guide for additional information
				IsInstantiatedOnEachOptimizationIteration	= true;
				N					= 12;
				X					= 30;
				Y					= 2;
				fast				= 12; //MACD Period fast
				slow				= 26; //MACD Period slow
				signal				= 9; //MACD Period signal
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
			UpdateAfterStopLoss();
			if(	Position.MarketPosition != MarketPosition.Long && 
				Position.MarketPosition != MarketPosition.Short) //NoPositionExists
			{				
				if(MACD(fast, slow, signal)[0] > MACD(fast, slow, signal).Avg[0]) //MACD line > Signal line
				{
					if(AfterStopLoss && Close[0] < PositionClosingPrice + Y*ATR(N)[0]) //AfterStopLoss && (Current Price < PositionClosingPrice + y*ATR(N))
					{
						Print("1: Close[0]" + Close[0] + " AfterStopLossBarrier, Plus: " + Y*ATR(N)[0]);
						return;
					}
					else
					{
						Print("Enter Long");
						ExitShort();
						EnterLong();
						PositionClosingPrice = Close[0];
						AfterStopLoss = true;
					}
				}
				else if(MACD(fast, slow, signal)[0] < MACD(fast, slow, signal).Avg[0]) //MACD line < Signal line
				{
					if(AfterStopLoss && Close[0] > PositionClosingPrice - Y*ATR(N)[0]) //AfterStopLoss && (Current Price > PositionClosingPrice - y*ATR(N))
					{
						Print("2: Close[0]" + Close[0] + " AfterStopLossBarrier, Minus: " + Y*ATR(N)[0]);
						return;
					}
					else
					{
						Print("Enter Short");
						ExitLong();
						EnterShort();
						PositionClosingPrice = Close[0];
						AfterStopLoss = true;
					}
				}
			}
		}
		
		private void UpdateAfterStopLoss()
		{
			if(Close[0] < AfterClosingPrice - Y*ATR(N)[0])
			{
				AfterStopLoss = false;
			}
			else if(Close[0] > AfterClosingPrice + Y*ATR(N)[0])
			{
				AfterStopLoss = false;
			}
		}
		
		protected override void OnPositionUpdate(Cbi.Position position, double averagePrice, int quantity, Cbi.MarketPosition marketPosition)
		{
		  if (position.MarketPosition == MarketPosition.Flat)
		  {
		    AfterStopLoss = true;
			AfterClosingPrice = Close[0];
			Print("ExitStopLoss: " + X*ATR(N)[0]);
			SetStopLoss(CalculationMode.Currency, X*ATR(N)[0]);
		  }
		  else
		  {
			Print("SetStopLoss: " + X*ATR(N)[0] + " Position: " + position.MarketPosition);
		  	SetStopLoss(CalculationMode.Currency, X*ATR(N)[0]);
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
