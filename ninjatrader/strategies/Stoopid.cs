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
		private Darvas Darv;
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
				Darv = Darvas();
				AddChartIndicator(Darv);
			}
		}

		protected override void OnBarUpdate()
		{
			if (!IsTradingTime())
			{
				ExitLong();
				ExitShort();
			}
			else if (Darv.BuySignal)
				EnterLong(TradeAmount);
			else if (Darv.SellSignal)
				EnterShort(TradeAmount);
			
		}
		
		protected override void OnPositionUpdate(Position position, double averagePrice, int quantity, MarketPosition marketPosition)
		{
			if (position.MarketPosition == MarketPosition.Flat)
			{
				Trade lastTrade = SystemPerformance.AllTrades[SystemPerformance.AllTrades.Count - 1];

				if (lastTrade.ProfitCurrency > 0)
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
		#endregion
	}
}
