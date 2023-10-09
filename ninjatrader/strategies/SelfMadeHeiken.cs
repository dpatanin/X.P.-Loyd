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
		private Sigmoid SigAvg;
		private Sigmoid SigAcc;
		private Sigmoid SigVel;
		private SigmoidGate Gate;
		
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
				Period = 5;
				Smooth = 2;
				SignalAvg = 10;
				SignalVel = 10;
				SignalAcc = 10;
				Threshold = 0.9;
			}
			else if (State == State.Configure && Category == Category.Optimize)
				IsInstantiatedOnEachOptimizationIteration = false;
			else if (State == State.DataLoaded)
			{
				TradeAmount = 1;
				Heiken = HeikenGrad(Period, Smooth);
				
				SigAvg = Sigmoid(Heiken.Avg, SignalAvg, Threshold, Brushes.Gold);
				SigVel = Sigmoid(Heiken, SignalVel, Threshold, Brushes.RoyalBlue);
				SigAcc = Sigmoid(Heiken.Pitch, SignalAcc, Threshold, Brushes.Violet);
				
				List<ISeries<double>> signals = new List<ISeries<double>>{SigAvg.Default, SigVel.Default, SigAcc.Default};
				Gate = SigmoidGate(signals, Threshold);
				
				AddChartIndicator(Heiken.Heiken);
				AddChartIndicator(Heiken);
				AddChartIndicator(SigAvg);
				AddChartIndicator(SigVel);
				AddChartIndicator(SigAcc);
				AddChartIndicator(Gate);
			}
		}

		protected override void OnBarUpdate()
		{			
			if (!IsTradingTime() || Gate[0] == 0)
			{
				ExitLong();
				ExitShort();
			}
			else if (Gate[0] == 1)
				EnterLong(TradeAmount);
			else if (Gate[0] == -1)
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
		[Display(Name = "SignalAvg", GroupName = "Parameters", Order = 3)]
		public double SignalAvg
		{ get; set; }

		[Range(0, int.MaxValue), NinjaScriptProperty]
		[Display(Name = "SignalVel", GroupName = "Parameters", Order = 4)]
		public double SignalVel
		{ get; set; }
		
		[Range(0, int.MaxValue), NinjaScriptProperty]
		[Display(Name = "SignalAcc", GroupName = "Parameters", Order = 5)]
		public double SignalAcc
		{ get; set; }		
		
		[Range(0, int.MaxValue), NinjaScriptProperty]
		[Display(Name = "Threshold", GroupName = "Parameters", Order = 6)]
		public double Threshold
		{ get; set; }
		#endregion
	}
}
