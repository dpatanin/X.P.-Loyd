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
		private int StopLossCount;
		
		private HeikenGrad Heiken;
		private Momentum Moment;
		private MFI Mfi;
		private ChaikinVolatility ChaiVol;
		private PFE Pfe;
		
		private Sigmoid SigAVG;
		private Sigmoid SigVEL;
		private Sigmoid SigMOM;
		private Sigmoid SigMFI;
		private Sigmoid SigPFE;
		private Sigmoid SigVOL;
		private SigmoidGate Activator;
		private SigmoidGate Inhibitor;
		
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
				Period = 14;
				Smooth = 3;
				Threshold = 0.9;
				Imperviousness = 2;
				StopLossCurr = 5;
				StopLossBreak = 5;
				
				SignalAVG = 25;
				SignalVEL = 15;
				SignalMOM = 15;
				SignalMFI = 5;
				SignalVOL = 0.15;
				SignalPFE = 7;
			}
			else if (State == State.Configure && Category == Category.Optimize)
				IsInstantiatedOnEachOptimizationIteration = false;
			else if (State == State.DataLoaded)
			{
				TradeAmount = 1;
				StopLossCount = 0;
				
				Heiken = HeikenGrad(Period, Smooth);
				Moment = Momentum(Heiken, Period);
				Mfi = MFI(Heiken, Period);
				ChaiVol = ChaikinVolatility(Heiken, Period, Period);
				Pfe = PFE(Heiken, Period, Smooth);
				
				SigAVG = Sigmoid(Heiken.Avg, SignalAVG, Threshold, 0, Brushes.Gold);
				SigVEL = Sigmoid(Heiken, SignalVEL, Threshold, 0, Brushes.RoyalBlue);
				SigMOM = Sigmoid(Moment, SignalMOM, Threshold,  0, Brushes.DarkCyan);
				SigMFI = Sigmoid(Mfi, SignalMFI, Threshold, -50, Brushes.Crimson);
				SigVOL = Sigmoid(ChaiVol, SignalVOL, Threshold, 0, Brushes.Crimson);
				SigPFE = Sigmoid(Pfe, SignalPFE, Threshold, 0, Brushes.SlateBlue);
				
				List<ISeries<double>> activeSignals = new List<ISeries<double>>{SigAVG.Default, SigVEL.Default, SigMFI.Default, SigMOM.Default, SigPFE.Default};
				List<ISeries<double>> inhibitorSignals = new List<ISeries<double>>{SigVOL.Default};
				Activator = SigmoidGate(activeSignals, Threshold, Imperviousness, Brushes.Turquoise);
				Inhibitor = SigmoidGate(inhibitorSignals, Threshold, Imperviousness, Brushes.Crimson);
				
				AddChartIndicator(Heiken.Heiken);
				//AddChartIndicator(ChaiVol);
				//AddChartIndicator(Moment);
				//AddChartIndicator(Heiken);
				//AddChartIndicator(Mfi);
				//AddChartIndicator(Pfe);
				//AddChartIndicator(SigAVG);
				//AddChartIndicator(SigVEL);
				//AddChartIndicator(SigMOM);
				//AddChartIndicator(SigMFI);
				//AddChartIndicator(SigVOL);
				//AddChartIndicator(SigPFE);
				AddChartIndicator(Inhibitor);
				AddChartIndicator(Activator);
			}
		}

		protected override void OnBarUpdate()
		{
			bool blockSignal = Inhibitor[0] != 0;
			
			if (Position.MarketPosition != MarketPosition.Flat)
			{
				double profitLoss = Position.GetUnrealizedProfitLoss(PerformanceUnit.Currency, Close[0]);
				if (profitLoss < -StopLossCurr)
					StopLossCount = 1;
			}
			
			if (!IsTradingTime() || Activator[0] == 0 || StopLossCount >= 1 && StopLossCount <= StopLossBreak)
			{
				ExitLong();
				ExitShort();
				
				if (StopLossCount > StopLossBreak)
					StopLossCount = 0;
				else if (StopLossCount >= 1)
					StopLossCount++;
			}
			else if (Activator[0] == 1 && !blockSignal)
				EnterLong(TradeAmount);
			else if (Activator[0] == -1 && !blockSignal)
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
		[Display(Name = "Threshold", GroupName = "Parameters", Order = 3)]
		public double Threshold
		{ get; set; }
		
		[Range(0, int.MaxValue), NinjaScriptProperty]
		[Display(Name = "Imperviousness", GroupName = "Parameters", Order = 4)]
		public double Imperviousness
		{ get; set; }
		
		[Range(0, int.MaxValue), NinjaScriptProperty]
		[Display(Name = "Stop Loss (Currency)", GroupName = "Parameters", Order = 5)]
		public double StopLossCurr
		{ get; set; }
		
		[Range(0, int.MaxValue), NinjaScriptProperty]
		[Display(Name = "Stop Break (Bars suration)", GroupName = "Parameters", Order = 6)]
		public double StopLossBreak
		{ get; set; }
		
		[Range(0, int.MaxValue), NinjaScriptProperty]
		[Display(Name = "SignalAVG", GroupName = "Amplifier", Order = 0)]
		public double SignalAVG
		{ get; set; }

		[Range(0, int.MaxValue), NinjaScriptProperty]
		[Display(Name = "SignalVEL", GroupName = "Amplifier", Order = 1)]
		public double SignalVEL
		{ get; set; }
		
		[Range(0, int.MaxValue), NinjaScriptProperty]
		[Display(Name = "SignalMOM", GroupName = "Amplifier", Order = 2)]
		public double SignalMOM
		{ get; set; }
		
		[Range(0, int.MaxValue), NinjaScriptProperty]
		[Display(Name = "SignalMFI", GroupName = "Amplifier", Order = 3)]
		public double SignalMFI
		{ get; set; }	
		
		[Range(0, int.MaxValue), NinjaScriptProperty]
		[Display(Name = "SignalVOL", GroupName = "Amplifier", Order = 4)]
		public double SignalVOL
		{ get; set; }
		
		[Range(0, int.MaxValue), NinjaScriptProperty]
		[Display(Name = "SignalPFE", GroupName = "Amplifier", Order = 5)]
		public double SignalPFE
		{ get; set; }
		#endregion
	}
}
