using Mandelbrot_Generator;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Mandelbrot_View_GL
{
    class Settings
    {
        public int ResolutionX { get; set; }
        public int ResolutionY { get; set; }
        public int RenderResolutionX { get; set; }
        public int RenderResolutionY { get; set; }
        public bool FullScreen { get; set; }

        public int SplitY { get; set; }

        public int OpenCL_DeviceID { get; set; }
        public string MaxiterMode { get; set; }

        public List<MyColor> Colors { get; set; }
        public List<double> Weight { get; set; }
    }
}
