using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Mandelbrot_Generator
{
    public static class MandelbrotHelper
    {
        public static void GetMandelbrotRegion(double x0, double x1, double y0, double y1, byte[] buffer, int res_x, int res_y, int max_iter, byte[] colors)
        {
            double mx = (x1 - x0) / res_x;
            double my = (y1 - y0) / res_y;
            Parallel.For(0, res_y, delegate (int y)
            //for (int y = 0; y < res_y; y++)
            {
                for (int x = 0; x < res_x; x++)
                {
                    double cx = x0 + x * mx;
                    double cy = y0 + y * my;
                    int v = MandelbrotCode.iterate(cx, cy, max_iter);

                    buffer[(y * res_x + x) * 4 + 0] = (byte)(colors[v * 3 + 0]);
                    buffer[(y * res_x + x) * 4 + 1] = (byte)(colors[v * 3 + 1]);
                    buffer[(y * res_x + x) * 4 + 2] = (byte)(colors[v * 3 + 2]);
                    buffer[(y * res_x + x) * 4 + 3] = 255;

                }
            });
        }
        public static byte[] GetLinearGradient(double v, double minV, double maxV, byte[] colors, double[] range)
        {
            int i = 0;
            double sum = range.Sum();
            double bas = (v - minV) / (maxV - minV) * sum;
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
        public static MyColor GetLinearGradient(double v, double minV, double maxV, MyColor[] colors, double[] range)
        {
            byte[] c = GetLinearGradient(v, minV, maxV, colors.Select(cl => cl.GetBytes_RGBA()).Aggregate((c1, c2) => c1.Concat(c2).ToArray()), range);
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

        public static void UpdateMandelbrotRegionGPUSplittedY(MandelbrotGPU gpu, double x, double y, double viewport, int maxiter, int res_x, int res_y, int split_y, int y_index)
        {
            double vx = viewport * res_x / res_y;
            double y_size = viewport * 2 / split_y;

            gpu.UpdateArea(
                x - vx,
                x + vx,
                y - viewport + y_size * (y_index + 0),
                y - viewport + y_size * (y_index + 1),
                maxiter);
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
