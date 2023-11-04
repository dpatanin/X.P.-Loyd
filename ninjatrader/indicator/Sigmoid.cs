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
	public class Sigmoid : Indicator
	{
		protected override void OnStateChange()
		{
			if (State == State.SetDefaults)
			{
				Description							= @"Calculates a Sigmoid activation curve";
				Name								= "Sigmoid";
				Calculate							= Calculate.OnEachTick;
				DrawOnPricePanel					= false;
				IsSuspendedWhileInactive			= false;
				BarsRequiredToPlot					= 1;
				
				Signal = 1;
				Threshold = 0.9;
				Offset = 0;
				UserDefinedBrush = Brushes.MediumVioletRed;
				AddPlot(UserDefinedBrush, "Sigmoid");
			}
			if (State == State.DataLoaded)
			{
				Plots[0].Brush = UserDefinedBrush;
				Draw.HorizontalLine(this, "Zero", 0, Brushes.WhiteSmoke);
				Draw.HorizontalLine(this, "Upper Threshold", Threshold, Brushes.DarkCyan);
				Draw.HorizontalLine(this, "Lower Threshold", 0-Threshold, Brushes.DarkCyan);
			}
		}

		protected override void OnBarUpdate()
		{
			Default[0] = -(2 / (1+Math.Exp(Signal * (Input[0]+Offset))) - 1);
		}

		#region Properties
		
		[Range(0, int.MaxValue), NinjaScriptProperty]
		[Display(Name = "Signal", GroupName = "Parameters", Order = 0)]
		public double Signal
		{ get; set; }
		
		[Range(0, 1), NinjaScriptProperty]
		[Display(Name = "Threshold", GroupName = "Parameters", Order = 1)]
		public double Threshold
		{ get; set; }
		
		[NinjaScriptProperty]
		[Display(Name = "Offset", GroupName = "Parameters", Order = 2)]
		public double Offset
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
		private Sigmoid[] cacheSigmoid;
		public Sigmoid Sigmoid(double signal, double threshold, double offset, Brush userDefinedBrush)
		{
			return Sigmoid(Input, signal, threshold, offset, userDefinedBrush);
		}

		public Sigmoid Sigmoid(ISeries<double> input, double signal, double threshold, double offset, Brush userDefinedBrush)
		{
			if (cacheSigmoid != null)
				for (int idx = 0; idx < cacheSigmoid.Length; idx++)
					if (cacheSigmoid[idx] != null && cacheSigmoid[idx].Signal == signal && cacheSigmoid[idx].Threshold == threshold && cacheSigmoid[idx].Offset == offset && cacheSigmoid[idx].UserDefinedBrush == userDefinedBrush && cacheSigmoid[idx].EqualsInput(input))
						return cacheSigmoid[idx];
			return CacheIndicator<Sigmoid>(new Sigmoid(){ Signal = signal, Threshold = threshold, Offset = offset, UserDefinedBrush = userDefinedBrush }, input, ref cacheSigmoid);
		}
	}
}

namespace NinjaTrader.NinjaScript.MarketAnalyzerColumns
{
	public partial class MarketAnalyzerColumn : MarketAnalyzerColumnBase
	{
		public Indicators.Sigmoid Sigmoid(double signal, double threshold, double offset, Brush userDefinedBrush)
		{
			return indicator.Sigmoid(Input, signal, threshold, offset, userDefinedBrush);
		}

		public Indicators.Sigmoid Sigmoid(ISeries<double> input , double signal, double threshold, double offset, Brush userDefinedBrush)
		{
			return indicator.Sigmoid(input, signal, threshold, offset, userDefinedBrush);
		}
	}
}

namespace NinjaTrader.NinjaScript.Strategies
{
	public partial class Strategy : NinjaTrader.Gui.NinjaScript.StrategyRenderBase
	{
		public Indicators.Sigmoid Sigmoid(double signal, double threshold, double offset, Brush userDefinedBrush)
		{
			return indicator.Sigmoid(Input, signal, threshold, offset, userDefinedBrush);
		}

		public Indicators.Sigmoid Sigmoid(ISeries<double> input , double signal, double threshold, double offset, Brush userDefinedBrush)
		{
			return indicator.Sigmoid(input, signal, threshold, offset, userDefinedBrush);
		}
	}
}

#endregion
