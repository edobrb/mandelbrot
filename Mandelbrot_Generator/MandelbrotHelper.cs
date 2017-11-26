using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Mandelbrot_Generator
{
    public static class MandelbrotHelper
    {
        public static byte[] GetLinearGradient(double v, double minV, double maxV, byte[] colors, double[] range, Func<double, double> gradient)
        {
            int i = 0;
            double sum = range.Sum();
            double bas = (v - minV) / (maxV - minV);
            bas = gradient(bas) * sum;
            while (i < range.Length - 1 && bas > range[i])
            {
                bas -= range[i];
                i++;
            }

            byte a = (byte)linear(bas, 0, range[i], colors[i * 4 + 0], colors[(i + 1) * 4 + 0]);
            byte r = (byte)linear(bas, 0, range[i], colors[i * 4 + 1], colors[(i + 1) * 4 + 1]);
            byte g = (byte)linear(bas, 0, range[i], colors[i * 4 + 2], colors[(i + 1) * 4 + 2]);
            byte b = (byte)linear(bas, 0, range[i], colors[i * 4 + 3], colors[(i + 1) * 4 + 3]);
           
            return new byte[] { a, r, g, b };
        }
        public static MyColor GetLinearGradient(double v, double minV, double maxV, MyColor[] colors, double[] range, Func<double, double> gradient)
        {
            byte[] c = GetLinearGradient(v, minV, maxV, colors.Select(cl => cl.GetBytes_RGBA()).Aggregate((c1, c2) => c1.Concat(c2).ToArray()), range, gradient);
            return new MyColor(c[0], c[1], c[2], c[3]);
        }
        public static double linear(double x, double x0, double x1, double y0, double y1)
        {
            if ((x1 - x0) == 0)
            {
                return (y0 + y1) / 2;
            }
            return y0 + (x - x0) * (y1 - y0) / (x1 - x0);
        }
    }
    public class MyColor
    {
        public byte A { get; set; }
        public byte R { get; set; }
        public byte G { get; set; }
        public byte B { get; set; }
        public MyColor(byte r, byte g, byte b, byte a)
        {
            R = r;
            G = g;
            B = b;
            A = a;
        }

        public uint GetRGBA()
        {
            uint v = A;
            v <<= 8;
            v += B;
            v <<= 8;
            v += G;
            v <<= 8;
            v += R;
            return v;
        }
        public byte[] GetBytes_ARGB()
        {
            return new byte[] { A, R, G, B };
        }
        public byte[] GetBytes_RGBA()
        {
            return new byte[] { R, G, B, A };
        }

        public static MyColor Black = new MyColor(0, 0, 0, 255);
        public static MyColor Red = new MyColor(255, 0, 0, 255);
        public static MyColor DarkRed = new MyColor(139, 0, 0, 255);
        public static MyColor DarkGray = new MyColor(169, 169, 169, 255);
    }
}
