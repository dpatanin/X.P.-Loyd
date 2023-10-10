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
using NinjaTrader.Data;
using NinjaTrader.NinjaScript;
using NinjaTrader.Core.FloatingPoint;
using NinjaTrader.NinjaScript.DrawingTools;
using SharpDX;
#endregion

//This namespace holds Indicators in this folder and is required. Do not change it. 
namespace NinjaTrader.NinjaScript.Indicators
{
	public class SigmoidGate : Indicator
	{
		private int QueuedOutput;
		private int QueuedCount;
		
		protected override void OnStateChange()
		{
			if (State == State.SetDefaults)
			{
				Description							= @"Receives an array of input series as outputs from sigmoid indicators.";
				Name								= "SigmoidGate";
				Calculate							= Calculate.OnEachTick;
				DrawOnPricePanel					= false;
				IsSuspendedWhileInactive			= false;
				BarsRequiredToPlot					= 1;
				
				Threshold = 0.9;
				Imperviousness = 0;
				
				UserDefinedBrush  = Brushes.Turquoise;
				AddPlot(UserDefinedBrush, "Signal");
			}
			if (State == State.DataLoaded)
			{
				QueuedCount = 0;
				QueuedOutput = 0;
				Plots[0].Brush = UserDefinedBrush;
				Draw.HorizontalLine(this, "Zero", 0, Brushes.WhiteSmoke);
			}
		}

		protected override void OnBarUpdate()
		{
			int countLong = 0;
			int countShort = 0;
			int countStay = 0;
			
			foreach (Series<double> signal in Signals)
			{
				if (signal[0] < -Threshold)
					countShort++;
				else if (signal[0] > Threshold)
					countLong++;
				else
					countStay++;
			}
			
			int output = 0;
			int highestCount = Math.Max(countLong, Math.Max(countShort, countStay));
			
			if (highestCount == countLong)
				output = 1;
			else if (highestCount == countShort)
				output = -1;
			
			if (CurrentBar < 1 || output == Default[1] || QueuedCount == Imperviousness)
			{
				Default[0] = output;
				QueuedOutput = output;
				QueuedCount = 0;
			}
			else if (output != Default[1] && output != QueuedOutput)
			{
				Default[0] = Default[1];
				QueuedOutput = output;
				QueuedCount = 0;
			}
			else if (output == QueuedOutput)
			{
				Default[0] = Default[1];
				QueuedCount++;
			}
		}

		#region Properties
		[NinjaScriptProperty]
		[Display(Name = "Signals", GroupName = "Parameters", Order = 0)]
		public List<ISeries<double>> Signals
		{ get; set; }
		
		[Range(0, 1), NinjaScriptProperty]
		[Display(Name = "Threshold", GroupName = "Parameters", Order = 1)]
		public double Threshold
		{ get; set; }
		
		[Range(0, int.MaxValue), NinjaScriptProperty]
		[Display(Name = "Imperviousness", GroupName = "Parameters", Order = 2)]
		public double Imperviousness
		{ get; set; }
		
		[NinjaScriptProperty]
		[XmlIgnore]
		[Display(Name="UserDefinedBrush", GroupName="Parameters", Order=3)]
		public Brush UserDefinedBrush
		{ get; set; }
		
		[Browsable(false)]
		public string UserDefinedBrushSerializable
		{
		get { return Serialize.BrushToString(UserDefinedBrush); }
		set { UserDefinedBrush = Serialize.StringToBrush(value); }
		}
		
		[Browsable(false)]
		[XmlIgnore()]
		public Series<double> Default
		{
			get { return Values[0]; }
		}
		#endregion
	
		}
}

#region NinjaScript generated code. Neither change nor remove.

namespace NinjaTrader.NinjaScript.Indicators
{
	public partial class Indicator : NinjaTrader.Gui.NinjaScript.IndicatorRenderBase
	{
		private SigmoidGate[] cacheSigmoidGate;
		public SigmoidGate SigmoidGate(List<ISeries<double>> signals, double threshold, double imperviousness, Brush userDefinedBrush)
		{
			return SigmoidGate(Input, signals, threshold, imperviousness, userDefinedBrush);
		}

		public SigmoidGate SigmoidGate(ISeries<double> input, List<ISeries<double>> signals, double threshold, double imperviousness, Brush userDefinedBrush)
		{
			if (cacheSigmoidGate != null)
				for (int idx = 0; idx < cacheSigmoidGate.Length; idx++)
					if (cacheSigmoidGate[idx] != null && cacheSigmoidGate[idx].Signals == signals && cacheSigmoidGate[idx].Threshold == threshold && cacheSigmoidGate[idx].Imperviousness == imperviousness && cacheSigmoidGate[idx].UserDefinedBrush == userDefinedBrush && cacheSigmoidGate[idx].EqualsInput(input))
						return cacheSigmoidGate[idx];
			return CacheIndicator<SigmoidGate>(new SigmoidGate(){ Signals = signals, Threshold = threshold, Imperviousness = imperviousness, UserDefinedBrush = userDefinedBrush }, input, ref cacheSigmoidGate);
		}
	}
}

namespace NinjaTrader.NinjaScript.MarketAnalyzerColumns
{
	public partial class MarketAnalyzerColumn : MarketAnalyzerColumnBase
	{
		public Indicators.SigmoidGate SigmoidGate(List<ISeries<double>> signals, double threshold, double imperviousness, Brush userDefinedBrush)
		{
			return indicator.SigmoidGate(Input, signals, threshold, imperviousness, userDefinedBrush);
		}

		public Indicators.SigmoidGate SigmoidGate(ISeries<double> input , List<ISeries<double>> signals, double threshold, double imperviousness, Brush userDefinedBrush)
		{
			return indicator.SigmoidGate(input, signals, threshold, imperviousness, userDefinedBrush);
		}
	}
}

namespace NinjaTrader.NinjaScript.Strategies
{
	public partial class Strategy : NinjaTrader.Gui.NinjaScript.StrategyRenderBase
	{
		public Indicators.SigmoidGate SigmoidGate(List<ISeries<double>> signals, double threshold, double imperviousness, Brush userDefinedBrush)
		{
			return indicator.SigmoidGate(Input, signals, threshold, imperviousness, userDefinedBrush);
		}

		public Indicators.SigmoidGate SigmoidGate(ISeries<double> input , List<ISeries<double>> signals, double threshold, double imperviousness, Brush userDefinedBrush)
		{
			return indicator.SigmoidGate(input, signals, threshold, imperviousness, userDefinedBrush);
		}
	}
}

#endregion
