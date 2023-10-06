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
	public class MACD_ATR : Strategy
	{
		private MACD Macd;
		private ATR Atr;
		private double AtrReference;
		private int TradeAmount;
		private int NumBarsInCurrentAtrSlide;
		
		protected override void OnStateChange()
		{
			if (State == State.SetDefaults)
			{
				Description = @"MACD based strategy with ATR exits.";
				Name = "MACD ATR";
				
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
				Fast = 1;
				Slow = 395;
				Signal = 435;
				
				// ATR related params
				AtrPeriod = 12;
				TakeProfitBarrieScale = 21;
				StopLossBarrierScale = 8;
				RecoveryBarrierScale = 5;
				
				EnableSliding = true;
				FocusStyle = FocusType.FractionalExp;
				FocusLimit = 0.1;
				FocusStrength = 0.1;
				MacdTrendFollow = 1;
				
				// Plots
				AddPlot(new Stroke(Brushes.Green, 2), PlotStyle.Dot, "ActiveLowerATR");
				AddPlot(new Stroke(Brushes.Green, 2), PlotStyle.Dot, "ActiveUpperATR");
				AddPlot(new Stroke(Brushes.Red, 2), PlotStyle.Dot, "RecoveryLowerATR");
				AddPlot(new Stroke(Brushes.Red, 2), PlotStyle.Dot, "RecoveryUpperATR");
				AddPlot(new Stroke(Brushes.Blue, DashStyleHelper.Dash, 2), PlotStyle.Line, "FocusLowerATR");
				AddPlot(new Stroke(Brushes.Blue, DashStyleHelper.Dash, 2), PlotStyle.Line, "FocusUpperATR");
			}
			else if (State == State.Configure)
			{	
				// Boosts performance in optimization mode
				if (Category == Category.Optimize)
					IsInstantiatedOnEachOptimizationIteration = false;
			}
			else if (State == State.DataLoaded)
			{				
				TradeAmount = 1;
				NumBarsInCurrentAtrSlide = 0;
				Macd = MACD(Fast, Slow, Signal);
				Atr = ATR(AtrPeriod);
				
				AddChartIndicator(Macd);
				AddChartIndicator(Atr);
			}
		}
		
		protected override void OnBarUpdate()
		{
			MarketPosition pos = Position.MarketPosition;
			if (!IsTradingTime())
			{
				ExitLong();
				ExitShort();
			}
			else if (pos != MarketPosition.Flat)
				HandleAtrExit();
			else
				HandleAtrEntry();
		}
		
		protected override void OnPositionUpdate(Position position, double averagePrice, int quantity, MarketPosition marketPosition)
		{
			if (position.MarketPosition == MarketPosition.Flat)
			{
				NumBarsInCurrentAtrSlide = 0;
				Trade lastTrade = SystemPerformance.AllTrades[SystemPerformance.AllTrades.Count - 1];

				if(lastTrade.ProfitCurrency > 0)
				   TradeAmount += WinStreakBonus;
				else
				   TradeAmount = 1;
			}
			else
				AtrReference = averagePrice;
		}

		private bool IsTradingTime()
		{
			int now = ToTime(Time[0]);
			return now >= ToTime(StartTime) && now <= ToTime(EndTime);
		}
		
		private void HandleAtrEntry()
		{
			double lastExitPrice;
			if (SystemPerformance.AllTrades.Count > 0)
				lastExitPrice = SystemPerformance.AllTrades[SystemPerformance.AllTrades.Count - 1].Exit.Price;
			else
				lastExitPrice = Close[CurrentBar];
				
			double recoveryAtr = RecoveryBarrierScale * Atr[0] * TickSize;
			double lowerBarrier = lastExitPrice - recoveryAtr;
			double upperBarrier = lastExitPrice + recoveryAtr;
			
			if (RecoveryBarrierScale > 0)
			{
				Values[2][0] = lowerBarrier;
				Values[3][0] = upperBarrier;
			}
			
			if (Macd[0] > Macd.Avg[0] && Close[0] >= upperBarrier)
				EnterLong(TradeAmount);
			else if (Macd[0] <= Macd.Avg[0] && Close[0] <= lowerBarrier)
				EnterShort(TradeAmount);
		}

		private bool HandleAtrExit()
		{
			NumBarsInCurrentAtrSlide++;
			bool isLong = Position.MarketPosition == MarketPosition.Long;
			
			double scaledMacdDiff = Macd.Diff[0] * MacdTrendFollow;
			double lowerBound = Atr[0] * TickSize * (isLong ? StopLossBarrierScale : TakeProfitBarrieScale);
			double upperBound = Atr[0] * TickSize * (isLong ? TakeProfitBarrieScale : StopLossBarrierScale);
						
			if (FocusStyle != FocusType.None)
			{
				Values[4][0] = AtrReference - lowerBound * FocusLimit + scaledMacdDiff;
				Values[5][0] = AtrReference + upperBound * FocusLimit + scaledMacdDiff;
				
				Tuple<double, double> focusedBounds = Focus(lowerBound, upperBound);
				lowerBound = focusedBounds.Item1;
				upperBound = focusedBounds.Item2;
			}
			
			double lowerBarrier = AtrReference - lowerBound + scaledMacdDiff;
			double upperBarrier = AtrReference + upperBound + scaledMacdDiff;
			
			if (EnableSliding && (isLong && Close[0] > upperBarrier || !isLong && Close[0] < lowerBarrier))
			{
				NumBarsInCurrentAtrSlide = 0;
				AtrReference = Close[0];
				lowerBarrier = AtrReference - lowerBound;
				upperBarrier = AtrReference + upperBound;
			}
			
			Values[0][0] = lowerBarrier;
			Values[1][0] = upperBarrier;
			
			if (Close[0] < lowerBarrier || Close[0] > upperBarrier)
			{
				ExitLong();
				ExitShort();
				return true;
			}
			
			return false;
		}
		
		private Tuple<double, double> Focus(double lowerBound, double upperBound)
		{	
			double percentage;
			switch (FocusStyle)
			{
				case FocusType.Linear:
					percentage = 1 - FocusStrength * NumBarsInCurrentAtrSlide;
					break;
				case FocusType.FractionalExp:
					percentage = FractExp(FocusStrength, NumBarsInCurrentAtrSlide);
					break;
				default:
					percentage = 1;
					break;
			}
			
			double focusedLowerBound = lowerBound * Math.Max(percentage, FocusLimit);
			double focusedUpperBound = upperBound * Math.Max(percentage, FocusLimit);

			return Tuple.Create(focusedLowerBound, focusedUpperBound);
		}

		private double FractExp(double k, double x)
		{
			return 2.0 / (1.0 + Math.Exp(k * x));
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
		[Display(Name = "Win Streak Bonus", Description="0 = trade only with 1 contract", GroupName = "Base Parameters", Order = 0)]
		public int WinStreakBonus
		{ get; set; }
		
		[Range(1, int.MaxValue), NinjaScriptProperty]
		[Display(Name = "Fast", GroupName = "Base Parameters", Order = 1)]
		public int Fast
		{ get; set; }

		[Range(1, int.MaxValue), NinjaScriptProperty]
		[Display(Name = "Slow", GroupName = "Base Parameters", Order = 2)]
		public int Slow
		{ get; set; }

		[Range(1, int.MaxValue), NinjaScriptProperty]
		[Display(Name = "Signal", GroupName = "Base Parameters", Order = 3)]
		public int Signal
		{ get; set; }
		
		[Range(1, int.MaxValue), NinjaScriptProperty]
		[Display(Name = "ATR Weight / Period", Description="Higher means lesser susceptible to volatility", GroupName = "ATR Parameters", Order = 0)]
		public int AtrPeriod
		{ get; set; }
		
		[Range(1, int.MaxValue), NinjaScriptProperty]
		[Display(Name = "Recovery Barrier Scale", GroupName = "ATR Parameters", Order = 1)]
		public int RecoveryBarrierScale
		{ get; set; }
		
		[Range(1, int.MaxValue), NinjaScriptProperty]
		[Display(Name = "Take Profit Barrier Scale", Description="The lower, the safer.", GroupName = "ATR Parameters", Order = 2)]
		public int TakeProfitBarrieScale
		{ get; set; }
		
		[Range(1, int.MaxValue), NinjaScriptProperty]
		[Display(Name = "Stop Loss Barrier Scale", Description="The lower, the safer.", GroupName = "ATR Parameters", Order = 3)]
		public int StopLossBarrierScale
		{ get; set; }
		
		[NinjaScriptProperty]
		[Display(Name = "Enable Sliding", Description="Slides active ATR barriers on crossing take profit instead of exiting", GroupName = "ATR Parameters", Order = 4)]
		public bool EnableSliding
		{ get; set; }
		
		[NinjaScriptProperty]
		[Display(Name = "Focus Style", GroupName = "ATR Parameters", Order = 5)]
		public FocusType FocusStyle
		{ get; set; }
		
		[Range(0, 1), NinjaScriptProperty]
		[Display(Name = "Focus Limit", Description = "Percentage limit to which focus converges", GroupName = "ATR Parameters", Order = 6)]
		public double FocusLimit
		{ get; set; }

		[Range(0, 1), NinjaScriptProperty]
		[Display(Name = "Focus Strength", Description = "Percentage rate at which focus converges", GroupName = "ATR Parameters", Order = 7)]
		public double FocusStrength
		{ get; set; }
		
		[Range(0, int.MaxValue), NinjaScriptProperty]
		[Display(Name = "MACD Trend follow", Description = "Following the MACD.Diff trend (0 = no following)", GroupName = "ATR Parameters", Order = 8)]
		public double MacdTrendFollow
		{ get; set; }
		#endregion
	}
	
	public enum FocusType
	{
		None,
		Linear,
		FractionalExp,
	}
}
