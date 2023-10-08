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
	public class SelfMadeHeiken : Strategy
	{
		private int TradeAmount;
		private HeikenGrad Heiken;
		private Sigmoid Sig;
		
		protected override void OnStateChange()
		{
			if (State == State.SetDefaults)
			{
				Description = @"Heiken Ashi Calculation but self made";
				Name = "SelfMadeHeiken";
				
				// NinjaTrader params
				Calculate = Calculate.OnBarClose;
				EntriesPerDirection = 1;
				EntryHandling = EntryHandling.AllEntries;
				IsExitOnSessionCloseStrategy = true;
				ExitOnSessionCloseSeconds = 930;
				IsFillLimitOnTouch = false;
				MaximumBarsLookBack = MaximumBarsLookBack.TwoHundredFiftySix;
				OrderFillResolution = OrderFillResolution.Standard;
				Slippage = 0;
				StartBehavior = StartBehavior.WaitUntilFlatSynchronizeAccount;
				TimeInForce = TimeInForce.Gtc;
				TraceOrders = false;
				RealtimeErrorHandling = RealtimeErrorHandling.StopCancelClose;
				StopTargetHandling = StopTargetHandling.PerEntryExecution;
				BarsRequiredToTrade = 0;
				IsInstantiatedOnEachOptimizationIteration = true;
				
				// Time Window
				StartTime = DateTime.Parse("00:00", System.Globalization.CultureInfo.InvariantCulture);
				EndTime = DateTime.Parse("22:40", System.Globalization.CultureInfo.InvariantCulture);
				
				// Base Params
				WinStreakBonus = 0;
				Period = 2;
				Smooth = 2;
				Signal = 1;
				Threshold = 0.5;
			}
			else if (State == State.Configure && Category == Category.Optimize)
				IsInstantiatedOnEachOptimizationIteration = false;
			else if (State == State.DataLoaded)
			{
				TradeAmount = 1;
				Heiken = HeikenGrad(Period, Smooth);
				Sig = Sigmoid(Heiken.Avg, Signal, Threshold);
				
				AddChartIndicator(CustomHeikenAshi());
				AddChartIndicator(Heiken);
				AddChartIndicator(Sig);
			}
		}

		protected override void OnBarUpdate()
		{
			bool longOpen = Sig[0] > -Threshold;
			bool shortOpen = Sig[0] < Threshold;
			
			bool accLong = Heiken.Pitch[0] > 0;
			bool accShort = Heiken.Pitch[0] < 0;
			bool velLong = Heiken[0] > 0;
			bool velShort = Heiken[0] < 0;
			bool steepAcc = Math.Abs(Heiken.Pitch[0]) > Math.Abs(Heiken[0]);
			
			bool steepLong = accLong && steepAcc;
			bool steepShort = accShort && steepAcc;
			bool shallowLong = accLong && velLong;
			bool shallowShort = accShort && velShort;
					
			if (!IsTradingTime())
			{
				ExitLong();
				ExitShort();
			}
			else if (longOpen && (steepShort || !steepLong && shallowShort || !shallowLong && velShort))
				EnterLong(TradeAmount);
			else if (shortOpen && (steepLong || !steepShort && shallowLong || !shallowShort && velLong))
				EnterShort(TradeAmount);		
		}
		
		protected override void OnPositionUpdate(Position position, double averagePrice, int quantity, MarketPosition marketPosition)
		{
			if (SystemPerformance.AllTrades.Count > 0)
			{
				Trade lastTrade = SystemPerformance.AllTrades[SystemPerformance.AllTrades.Count - 1];

				if(lastTrade.ProfitCurrency > 0)
				   TradeAmount += WinStreakBonus;
				else
				   TradeAmount = 1;
			}
		}

		private bool IsTradingTime()
		{
			int now = ToTime(Time[0]);
			return now >= ToTime(StartTime) && now <= ToTime(EndTime);
		}

		#region Properties
		[NinjaScriptProperty]
		[PropertyEditor("NinjaTrader.Gui.Tools.TimeEditorKey")]
		[Display(Name="Start Time", GroupName="Time Window", Order=0)]
		public DateTime StartTime
		{ get; set; }
		
		[NinjaScriptProperty]
		[PropertyEditor("NinjaTrader.Gui.Tools.TimeEditorKey")]
		[Display(Name="End Time", GroupName="Time Window", Order=1)]
		public DateTime EndTime
		{ get; set; }
		
		[Range(0, int.MaxValue), NinjaScriptProperty]
		[Display(Name = "Win Streak Bonus", Description="0 = trade only with 1 contract", GroupName = "Parameters", Order = 0)]
		public int WinStreakBonus
		{ get; set; }
		
		[Range(1, int.MaxValue), NinjaScriptProperty]
		[Display(Name = "Period", GroupName = "Parameters", Order = 1)]
		public int Period
		{ get; set; }
		
		[Range(1, int.MaxValue), NinjaScriptProperty]
		[Display(Name = "Smooth", GroupName = "Parameters", Order = 2)]
		public int Smooth
		{ get; set; }
		
		[Range(0, int.MaxValue), NinjaScriptProperty]
		[Display(Name = "Signal", GroupName = "Parameters", Order = 3)]
		public double Signal
		{ get; set; }
		
		[Range(0, int.MaxValue), NinjaScriptProperty]
		[Display(Name = "Threshold", GroupName = "Parameters", Order = 4)]
		public double Threshold
		{ get; set; }
		#endregion
	}
}
