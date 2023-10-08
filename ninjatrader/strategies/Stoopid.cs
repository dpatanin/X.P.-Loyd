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
	public class Stoopid : Strategy
	{
		private EMA EClose;
		private EMA EOpen;
		private int TradeAmount;
		
		protected override void OnStateChange()
		{
			if (State == State.SetDefaults)
			{
				Description = @"Stoopid";
				Name = "Stoopid";
				
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
				GradientBars = 9;
				Period = 31;
				StopLossCurrency = 30;
				RecoverPeriod = 5;
			}
			else if (State == State.Configure)
			{	
				// Boosts performance in optimization mode
				if (Category == Category.Optimize)
					IsInstantiatedOnEachOptimizationIteration = false;
				
				SetStopLoss(CalculationMode.Currency, StopLossCurrency);
			}
			else if (State == State.DataLoaded)
			{				
				TradeAmount = 1;
				EClose = EMA(Close, Period);
				EOpen = EMA(Open, Period);
				
				AddChartIndicator(EClose);
				AddChartIndicator(EOpen);
			}
		}

		protected override void OnBarUpdate()
		{
			double grad = Gradient();
			int numTrades = SystemPerformance.AllTrades.Count;
			Trade lastTrade = numTrades >= 1 ? SystemPerformance.AllTrades[numTrades-1] : null;
			
			if (!IsTradingTime())
			{
				ExitLong();
				ExitShort();
			}
			else if (grad >= 0 && (lastTrade == null || lastTrade.Entry.MarketPosition != MarketPosition.Short))
				EnterShort(TradeAmount);
			else if (lastTrade == null || lastTrade.Entry.MarketPosition != MarketPosition.Long)
				EnterLong(TradeAmount);
		}
		
		protected override void OnPositionUpdate(Position position, double averagePrice, int quantity, MarketPosition marketPosition)
		{
			if (position.MarketPosition == MarketPosition.Flat)
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
		
		private bool IsRecovering()
		{
			int numBars = BarsSinceExitExecution();
			return numBars <= RecoverPeriod && numBars >= 0;
		}
		
		
		/// <summary>
		/// Calculates overall gradient over a given period.
		/// </summary>
		private double Gradient()
		{
			double gradient = 0;
			int gradientBars = Math.Min(GradientBars, CurrentBar+1);
			
			foreach (int i in Enumerable.Range(0, gradientBars-1))
				gradient += EClose[i] - EOpen[i];
			
			return gradient;
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
		[Display(Name = "GradientBars", GroupName = "Parameters", Order = 1)]
		public int GradientBars
		{ get; set; }

		[Range(1, int.MaxValue), NinjaScriptProperty]
		[Display(Name = "Period", GroupName = "Parameters", Order = 2)]
		public int Period
		{ get; set; }
		
		[Range(5, int.MaxValue), NinjaScriptProperty]
		[Display(Name = "Stop Loss Currency", GroupName = "Parameters", Order = 3)]
		public double StopLossCurrency
		{ get; set; }
		
		[Range(1, int.MaxValue), NinjaScriptProperty]
		[Display(Name = "Recover Period", GroupName = "Parameters", Order = 4)]
		public int RecoverPeriod
		{ get; set; }
		#endregion
	}
}
