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
	public class EMACrossOver : Strategy
	{
		private EMA emaFast;
		private SMA smaSlow;
		private int contracts;
		private Trade lastTrade;
		private int tradesCount;

		protected override void OnStateChange()
		{
			if (State == State.SetDefaults)
			{
				Description			= "";
				Name				= "EMA CrossOver";
				Fast				= 5;
				Slow				= 25;
				StopLoss			= 6;
				contracts			= 1;
				winstreakScale 		= 1;
				// This strategy has been designed to take advantage of performance gains in Strategy Analyzer optimizations
				// See the Help Guide for additional information
				IsInstantiatedOnEachOptimizationIteration = false;
			}
			else if (State == State.DataLoaded)
			{
				emaFast = EMA(Fast);
				smaSlow = SMA(Slow);

				emaFast.Plots[0].Brush = Brushes.Goldenrod;
				smaSlow.Plots[0].Brush = Brushes.SeaGreen;

				AddChartIndicator(emaFast);
				AddChartIndicator(smaSlow);
			}
			else if (State == State.Configure)
			{
    			//SetStopLoss(CalculationMode.Ticks, StopLoss);
			}
		}

		protected override void OnBarUpdate()
		{
			if (CurrentBar < BarsRequiredToTrade)
				return;

			if (SystemPerformance.AllTrades.Count > 1 && isWinstreakEnabled)
				lastTrade = SystemPerformance.AllTrades[SystemPerformance.AllTrades.Count - 1];

			if (lastTrade != null
				&& isWinstreakEnabled
				&& SystemPerformance.AllTrades.TradesCount > tradesCount)
			{
				if(lastTrade.ProfitCurrency > 0)
					contracts += winstreakScale;

				else if(lastTrade.ProfitCurrency < 0)
					contracts = 1;

				tradesCount++;
			}

			if (CrossAbove(emaFast, smaSlow, 1))
				EnterShort(contracts);
			else if (CrossBelow(emaFast, smaSlow, 1))
				EnterLong(contracts);
		}

		#region Properties
		[Range(1, int.MaxValue), NinjaScriptProperty]
		[Display(ResourceType = typeof(Custom.Resource), Name = "Fast", GroupName = "Parameters", Order = 0)]
		public int Fast
		{ get; set; }

		[Range(1, int.MaxValue), NinjaScriptProperty]
		[Display(ResourceType = typeof(Custom.Resource), Name = "Slow", GroupName = "Parameters", Order = 1)]
		public int Slow
		{ get; set; }

		[NinjaScriptProperty]
		[Range(1, int.MaxValue)]
		[Display(Name="StopLoss", Order=2, GroupName="Parameters")]
		public int StopLoss
		{ get; set; }

		[NinjaScriptProperty]
		[Display(Name="enable/disable Winstreak", Order=3, GroupName="Parameters")]
		public bool isWinstreakEnabled
		{ get; set; }

		[NinjaScriptProperty]
		[Range(1, int.MaxValue)]
		[Display(Name="Winstreak Scale (linear)", Order=4, GroupName="Parameters")]
		public int winstreakScale
		{ get; set; }

		#endregion
	}
}
