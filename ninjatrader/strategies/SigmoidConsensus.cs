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
	public class SigmoidConsensus : Strategy
	{
		private int StopLossCount;
		private double CurrentBestPrice;
		
		#region Indicators
		// Inhibitor
		private ChaikinVolatility ChaiVol;
		private ChoppinessIndex Chop;
		private RSquared R2;
		
		private Sigmoid SigVOL;
		private Sigmoid SigCHOP;
		private Sigmoid SigR2;
		
		// Activator
		private Momentum Moment;
		private MFI Mfi;
		private PFE Pfe;
		private BOP Bop;
		private DMI Dmi;
		private DoubleStochastics DStoch;
		private EaseOfMovement Eom;
		private FisherTransform Fish;
		private FOSC Fosc;
		private LinRegSlope LSlope;
		private MoneyFlowOscillator MFOsc;
		private PsychologicalLine Psy;
		private RSS Rss;
		private RVI Rvi;
		private TSI Tsi;
		private UltimateOscillator Ult;
		
		private Sigmoid SigMOM;
		private Sigmoid SigMFI;
		private Sigmoid SigPFE;
		private Sigmoid SigBOP;
		private Sigmoid SigDMI;
		private Sigmoid SigDSTOCH;
		private Sigmoid SigEOM;
		private Sigmoid SigFISH;
		private Sigmoid SigFOSC;
		private Sigmoid SigLSLOPE;
		private Sigmoid SigMFOSC;
		private Sigmoid SigPSY;
		private Sigmoid SigRSS;
		private Sigmoid SigRVI;
		private Sigmoid SigTSI;
		private Sigmoid SigULT;
		#endregion
		
		// Gates
		private SigmoidGate Activator;
		private SigmoidGate Inhibitor;
		
		protected override void OnStateChange()
		{
			if (State == State.SetDefaults)
			{
				Description = @"Consensus of several indicators translated utilizing sigmoid transformation.";
				Name = "SigmoidConsensus";
				
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
				TradeAmount 	= 1;
				WinStreakBonus 	= 0;
				Period 			= 14;
				Smooth 			= 3;
				Threshold 		= 0.9;
				Imperviousness 	= 2;
				StopLoss	 	= 9;
				StopLossBreak 	= 2;
				
				AddPlot(new Stroke(Brushes.OrangeRed, DashStyleHelper.Dash, 2), PlotStyle.Line, "StopLoss");
				
				#region Indicators
				// Inhibitor
				SignalVOL 		= 0.15;
				SignalCHOP 		= 0.04;
				SignalR2 		= 2;
				
				UseVOL 			= false;
				UseCHOP			= false;
				UseR2 			= true;
				
				// Activator
				SignalMOM 		= 4;
				SignalMFI 		= 4;
				SignalPFE 		= 8;
				SignalBOP 		= 5;
				SignalDMI 		= 5;
				SignalDSTOCH 	= 1;
				SignalEOM 		= 0.01;
				SignalFISH 		= 5;
				SignalFOSC 		= 16;
				SignalLSLOPE 	= 10;
				SignalMFOSC 	= 15;
				SignalPSY 		= 0.06;
				SignalRSS 		= 0.36;
				SignalRVI 		= 0.49;
				SignalTSI 		= 0.08;
				SignalULT 		= 0.94;
				
				UseMOM 			= false;
				UseMFI 			= false;
				UsePFE 			= false;
				UseBOP 			= false;
				UseDMI 			= true;
				UseDSTOCH 		= true;
				UseEOM 			= false;
				UseFISH 		= false;
				UseFOSC 		= true;
				UseLSLOPE 		= false;
				UseMFOSC 		= false;
				UsePSY 			= false;
				UseRSS 			= true;
				UseRVI 			= true;
				UseTSI 			= false;
				UseULT 			= false;
				#endregion
			}
			else if (State == State.Configure && Category == Category.Optimize)
				IsInstantiatedOnEachOptimizationIteration = false;
			else if (State == State.DataLoaded)
			{
				StopLossCount = 0;
				CurrentBestPrice = 0;
				
				List<ISeries<double>> activeSignals = InitActivationIndicators();
				List<ISeries<double>> inhibitorSignals = InitInhibitionIndicators();
				
				Activator = SigmoidGate(activeSignals, Threshold, Imperviousness, Brushes.Turquoise);
				Inhibitor = SigmoidGate(inhibitorSignals, Threshold, Imperviousness, Brushes.Crimson);

				AddChartIndicator(Inhibitor);
				AddChartIndicator(Activator);
			}
		}
		
		#region Init Indicators
		private List<ISeries<double>> InitActivationIndicators()
		{
			List<ISeries<double>> activeSignals = new List<ISeries<double>>();

			if (UseMOM)
			{
				Moment = Momentum(Period);
				SigMOM = Sigmoid(Moment, SignalMOM, Threshold,  0, Brushes.DarkCyan);
				activeSignals.Add(SigMOM.Default);
			}
			if (UseMFI)
			{
				Mfi = MFI(Period);
				SigMFI = Sigmoid(Mfi, SignalMFI, Threshold, -50, Brushes.Crimson);
				activeSignals.Add(SigMFI.Default);
			}
			if (UsePFE)
			{
				Pfe = PFE(Period, Smooth);
				SigPFE = Sigmoid(Pfe, SignalPFE, Threshold, 0, Brushes.SlateBlue);
				activeSignals.Add(SigPFE.Default);
			}
			if (UseBOP)
			{
				Bop = BOP(Smooth);
				SigBOP = Sigmoid(Bop, SignalBOP, Threshold, 0, Brushes.Yellow);
				activeSignals.Add(SigBOP.Default);
			}
			if (UseDMI)
			{
				Dmi = DMI(Period);
				SigDMI = Sigmoid(Dmi, SignalDMI, Threshold, 0, Brushes.Yellow);
				activeSignals.Add(SigDMI.Default);
			}
			if (UseDSTOCH)
			{
				DStoch = DoubleStochastics(Period);
				SigDSTOCH = Sigmoid(DStoch, SignalDSTOCH, Threshold, -50, Brushes.Crimson);
				activeSignals.Add(SigDSTOCH.Default);
			}
			if (UseEOM)
			{
				Eom = EaseOfMovement(Period, 10000);
				SigEOM = Sigmoid(Eom, SignalEOM, Threshold, 0, Brushes.SkyBlue);
				activeSignals.Add(SigEOM.Default);
			}
			if (UseFISH)
			{
				Fish = FisherTransform(Period);
				SigFISH = Sigmoid(Fish, SignalFISH, Threshold, 0, Brushes.SkyBlue);
				activeSignals.Add(SigFISH.Default);
			}
			if (UseFOSC)
			{
				Fosc = FOSC(Period);
				SigFOSC = Sigmoid(Fosc, SignalFOSC, Threshold, 0, Brushes.Cyan);
				activeSignals.Add(SigFOSC.Default);
			}
			if (UseLSLOPE)
			{
				LSlope = LinRegSlope(Period);
				SigLSLOPE = Sigmoid(LSlope, SignalLSLOPE, Threshold, 0, Brushes.Yellow);
				activeSignals.Add(SigLSLOPE.Default);
			}
			if (UseMFOSC)
			{
				MFOsc = MoneyFlowOscillator(Period);
				SigMFOSC = Sigmoid(MFOsc, SignalMFOSC, Threshold, 0, Brushes.Blue);
				activeSignals.Add(SigMFOSC.Default);
			}
			if (UsePSY)
			{
				Psy = PsychologicalLine(Period);
				SigPSY = Sigmoid(Psy, SignalPSY, Threshold, -50, Brushes.RoyalBlue);
				activeSignals.Add(SigPSY.Default);
			}
			if (UseRSS)
			{
				Rss = RSS(Smooth*3, Smooth*12, Period);
				SigRSS = Sigmoid(Rss, SignalRSS, Threshold, -50, Brushes.Cyan);
				activeSignals.Add(SigRSS.Default);
			}
			if (UseRVI)
			{
				Rvi = RVI(Period);
				SigRVI = Sigmoid(Rvi, SignalRVI, Threshold, -50, Brushes.Blue);
				activeSignals.Add(SigRVI.Default);
			}
			if (UseTSI)
			{
				Tsi = TSI(Smooth, Period);
				SigTSI = Sigmoid(Tsi, SignalTSI, Threshold, 0, Brushes.DarkCyan);
				activeSignals.Add(SigTSI.Default);
			}
			if (UseULT)
			{
				Ult = UltimateOscillator((int) Math.Ceiling(Period*0.5), Period, Period*2);
				SigULT = Sigmoid(Ult, SignalULT, Threshold, -50, Brushes.BlueViolet);
				activeSignals.Add(SigULT.Default);
			}
			
			return activeSignals;
		}
		
		private List<ISeries<double>> InitInhibitionIndicators()
		{
			List<ISeries<double>> inhibitorSignals = new List<ISeries<double>>();
			
			if (UseVOL)
			{
				ChaiVol = ChaikinVolatility(Period, Period);
				SigVOL = Sigmoid(ChaiVol, SignalVOL, Threshold, 0, Brushes.Crimson);
				inhibitorSignals.Add(SigVOL.Default);
			}
			if (UseCHOP)
			{
				Chop = ChoppinessIndex(Period);
				SigCHOP = Sigmoid(Chop, SignalCHOP, Threshold, 100, Brushes.Red);
				inhibitorSignals.Add(SigCHOP.Default);
			}
			if (UseVOL)
			{
				R2 = RSquared(Period);
				SigR2 = Sigmoid(R2, SignalR2, Threshold, -1, Brushes.Salmon);
				inhibitorSignals.Add(SigR2.Default);
			}
			
			return inhibitorSignals;
		}
		#endregion

		protected override void OnBarUpdate()
		{
			MarketPosition pos = Position.MarketPosition;
			if (pos != MarketPosition.Flat)
				UpdateStopLoss(pos);
			bool blockSignal = Inhibitor[0] != 0;
			
			if (!IsTradingTime() || Activator[0] == 0 || StopLossCount >= 1 && StopLossCount <= StopLossBreak)
			{
				ExitLong();
				ExitShort();
				
				if (StopLossCount > StopLossBreak)
					StopLossCount = 0;
				else if (StopLossCount >= 1)
				{	
					StopLossCount++;
					BackBrush = Brushes.Gray;
				}
			}
			else if (Activator[0] == 1 && !blockSignal)
				EnterLong(TradeAmount);
			else if (Activator[0] == -1 && !blockSignal)
				EnterShort(TradeAmount);		
			
		}
		
		protected override void OnPositionUpdate(Position position, double averagePrice, int quantity, MarketPosition marketPosition)
		{
			if (Position.MarketPosition == MarketPosition.Flat)
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
			else
				CurrentBestPrice = Position.AveragePrice;
		}

		private bool IsTradingTime()
		{
			int now = ToTime(Time[0]);
			return now >= ToTime(StartTime) && now <= ToTime(EndTime);
		}
		
		private void UpdateStopLoss(MarketPosition pos)
		{
			if (pos == MarketPosition.Long && Close[0] >= CurrentBestPrice || pos == MarketPosition.Short && Close[0] <= CurrentBestPrice)
				CurrentBestPrice = Close[0];
			else if (Math.Abs(CurrentBestPrice - Close[0]) >= StopLoss)
				StopLossCount = 1;
			
			Value[0] = CurrentBestPrice + (StopLoss * (pos == MarketPosition.Long ? -1 : 1));
		}

		#region Parameters
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
		[Display(Name = "TradeAmount (Base)", GroupName = "Parameters", Order = 0)]
		public int TradeAmount
		{ get; set; }
		
		[Range(0, int.MaxValue), NinjaScriptProperty]
		[Display(Name = "Win Streak Bonus", Description="0 = trade only with base amount", GroupName = "Parameters", Order = 1)]
		public int WinStreakBonus
		{ get; set; }
		
		[Range(1, int.MaxValue), NinjaScriptProperty]
		[Display(Name = "Period", GroupName = "Parameters", Order = 2)]
		public int Period
		{ get; set; }
		
		[Range(1, int.MaxValue), NinjaScriptProperty]
		[Display(Name = "Smooth", GroupName = "Parameters", Order = 3)]
		public int Smooth
		{ get; set; }
		
		[Range(0, int.MaxValue), NinjaScriptProperty]
		[Display(Name = "Threshold", GroupName = "Parameters", Order = 4)]
		public double Threshold
		{ get; set; }
		
		[Range(0, int.MaxValue), NinjaScriptProperty]
		[Display(Name = "Imperviousness", GroupName = "Parameters", Order = 5)]
		public double Imperviousness
		{ get; set; }
		
		[Range(0, int.MaxValue), NinjaScriptProperty]
		[Display(Name = "Stop Loss (Price diff)", GroupName = "Parameters", Order = 6)]
		public double StopLoss
		{ get; set; }
		
		[Range(0, int.MaxValue), NinjaScriptProperty]
		[Display(Name = "Stop Break (Bars duration)", GroupName = "Parameters", Order = 7)]
		public double StopLossBreak
		{ get; set; }
		#endregion
		
		#region Amplifier (Activator)
		[Range(0, int.MaxValue), NinjaScriptProperty]
		[Display(Name = "Use Momentum", GroupName = "Amplifier (Activator)", Order = 0)]
		public bool UseMOM
		{ get; set; }
		
		[Range(0, int.MaxValue), NinjaScriptProperty]
		[Display(Name = "SignalMOM", GroupName = "Amplifier (Activator)", Order = 1)]
		public double SignalMOM
		{ get; set; }
		
		[Range(0, int.MaxValue), NinjaScriptProperty]
		[Display(Name = "Use MFI", GroupName = "Amplifier (Activator)", Order = 2)]
		public bool UseMFI
		{ get; set; }
		
		[Range(0, int.MaxValue), NinjaScriptProperty]
		[Display(Name = "SignalMFI", GroupName = "Amplifier (Activator)", Order = 3)]
		public double SignalMFI
		{ get; set; }
		
		[Range(0, int.MaxValue), NinjaScriptProperty]
		[Display(Name = "Use PFE", GroupName = "Amplifier (Activator)", Order = 4)]
		public bool UsePFE
		{ get; set; }
		
		[Range(0, int.MaxValue), NinjaScriptProperty]
		[Display(Name = "SignalPFE", GroupName = "Amplifier (Activator)", Order = 5)]
		public double SignalPFE
		{ get; set; }
		
		[Range(0, int.MaxValue), NinjaScriptProperty]
		[Display(Name = "Use BOP", GroupName = "Amplifier (Activator)", Order = 6)]
		public bool UseBOP
		{ get; set; }
		
		[Range(0, int.MaxValue), NinjaScriptProperty]
		[Display(Name = "SignalBOP", GroupName = "Amplifier (Activator)", Order = 7)]
		public double SignalBOP
		{ get; set; }
		
		[Range(0, int.MaxValue), NinjaScriptProperty]
		[Display(Name = "Use DMI", GroupName = "Amplifier (Activator)", Order = 8)]
		public bool UseDMI
		{ get; set; }
		
		[Range(0, int.MaxValue), NinjaScriptProperty]
		[Display(Name = "SignalDMI", GroupName = "Amplifier (Activator)", Order = 9)]
		public double SignalDMI
		{ get; set; }
		
		[Range(0, int.MaxValue), NinjaScriptProperty]
		[Display(Name = "Use Double Stochastics", GroupName = "Amplifier (Activator)", Order = 10)]
		public bool UseDSTOCH
		{ get; set; }
		
		[Range(0, int.MaxValue), NinjaScriptProperty]
		[Display(Name = "SignalDSTOCH", GroupName = "Amplifier (Activator)", Order = 11)]
		public double SignalDSTOCH
		{ get; set; }
		
		[Range(0, int.MaxValue), NinjaScriptProperty]
		[Display(Name = "Use Ease of Movement", GroupName = "Amplifier (Activator)", Order = 12)]
		public bool UseEOM
		{ get; set; }
		
		[Range(0, int.MaxValue), NinjaScriptProperty]
		[Display(Name = "SignalEOM", GroupName = "Amplifier (Activator)", Order = 13)]
		public double SignalEOM
		{ get; set; }
		
		[Range(0, int.MaxValue), NinjaScriptProperty]
		[Display(Name = "Use Fisher transform", GroupName = "Amplifier (Activator)", Order = 14)]
		public bool UseFISH
		{ get; set; }
		
		[Range(0, int.MaxValue), NinjaScriptProperty]
		[Display(Name = "SignalFISH", GroupName = "Amplifier (Activator)", Order = 15)]
		public double SignalFISH
		{ get; set; }
		
		[Range(0, int.MaxValue), NinjaScriptProperty]
		[Display(Name = "Use Forecast Osc.", GroupName = "Amplifier (Activator)", Order = 16)]
		public bool UseFOSC
		{ get; set; }
		
		[Range(0, int.MaxValue), NinjaScriptProperty]
		[Display(Name = "SignalFOSC", GroupName = "Amplifier (Activator)", Order = 17)]
		public double SignalFOSC
		{ get; set; }
		
		[Range(0, int.MaxValue), NinjaScriptProperty]
		[Display(Name = "Use Lin. R. Slope", GroupName = "Amplifier (Activator)", Order = 18)]
		public bool UseLSLOPE
		{ get; set; }
		
		[Range(0, int.MaxValue), NinjaScriptProperty]
		[Display(Name = "SignalLSLOPE", GroupName = "Amplifier (Activator)", Order = 19)]
		public double SignalLSLOPE
		{ get; set; }
		
		[Range(0, int.MaxValue), NinjaScriptProperty]
		[Display(Name = "Use Money Flow Osc.", GroupName = "Amplifier (Activator)", Order = 20)]
		public bool UseMFOSC
		{ get; set; }
		
		[Range(0, int.MaxValue), NinjaScriptProperty]
		[Display(Name = "SignalMFOSC", GroupName = "Amplifier (Activator)", Order = 21)]
		public double SignalMFOSC
		{ get; set; }
		
		[Range(0, int.MaxValue), NinjaScriptProperty]
		[Display(Name = "Use Psychological Line", GroupName = "Amplifier (Activator)", Order = 22)]
		public bool UsePSY
		{ get; set; }
		
		[Range(0, int.MaxValue), NinjaScriptProperty]
		[Display(Name = "SignalPSY", GroupName = "Amplifier (Activator)", Order = 23)]
		public double SignalPSY
		{ get; set; }
		
		[Range(0, int.MaxValue), NinjaScriptProperty]
		[Display(Name = "Use RSS", GroupName = "Amplifier (Activator)", Order = 24)]
		public bool UseRSS
		{ get; set; }
		
		[Range(0, int.MaxValue), NinjaScriptProperty]
		[Display(Name = "SignalRSS", GroupName = "Amplifier (Activator)", Order = 25)]
		public double SignalRSS
		{ get; set; }
		
		[Range(0, int.MaxValue), NinjaScriptProperty]
		[Display(Name = "Use RVI", GroupName = "Amplifier (Activator)", Order = 26)]
		public bool UseRVI
		{ get; set; }
		
		[Range(0, int.MaxValue), NinjaScriptProperty]
		[Display(Name = "SignalRVI", GroupName = "Amplifier (Activator)", Order = 27)]
		public double SignalRVI
		{ get; set; }
		
		[Range(0, int.MaxValue), NinjaScriptProperty]
		[Display(Name = "Use TSI", GroupName = "Amplifier (Activator)", Order = 28)]
		public bool UseTSI
		{ get; set; }
		
		[Range(0, int.MaxValue), NinjaScriptProperty]
		[Display(Name = "SignalTSI", GroupName = "Amplifier (Activator)", Order = 29)]
		public double SignalTSI
		{ get; set; }
		
		[Range(0, int.MaxValue), NinjaScriptProperty]
		[Display(Name = "Use Ultimate Osc.", GroupName = "Amplifier (Activator)", Order = 30)]
		public bool UseULT
		{ get; set; }
		
		[Range(0, int.MaxValue), NinjaScriptProperty]
		[Display(Name = "SignalULT", GroupName = "Amplifier (Activator)", Order = 31)]
		public double SignalULT
		{ get; set; }
		#endregion
		
		#region Amplifier (Inhibitor)
		[Range(0, int.MaxValue), NinjaScriptProperty]
		[Display(Name = "Use Chaikan Volatility", GroupName = "Amplifier (Inhibitor)", Order = 0)]
		public bool UseVOL
		{ get; set; }
		
		[Range(0, int.MaxValue), NinjaScriptProperty]
		[Display(Name = "SignalVOL", GroupName = "Amplifier (Inhibitor)", Order = 1)]
		public double SignalVOL
		{ get; set; }
		
		[Range(0, int.MaxValue), NinjaScriptProperty]
		[Display(Name = "Use Choppiness Index", GroupName = "Amplifier (Inhibitor)", Order = 2)]
		public bool UseCHOP
		{ get; set; }
		
		[Range(0, int.MaxValue), NinjaScriptProperty]
		[Display(Name = "SignalCHOP", GroupName = "Amplifier (Inhibitor)", Order = 3)]
		public double SignalCHOP
		{ get; set; }
		
		[Range(0, int.MaxValue), NinjaScriptProperty]
		[Display(Name = "Use RSquared", GroupName = "Amplifier (Inhibitor)", Order = 4)]
		public bool UseR2
		{ get; set; }
		
		[Range(0, int.MaxValue), NinjaScriptProperty]
		[Display(Name = "SignalR2", GroupName = "Amplifier (Inhibitor)", Order = 5)]
		public double SignalR2
		{ get; set; }
		#endregion
	}
}
