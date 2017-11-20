using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Mandelbrot
{
    public static class Mandelbrot
    {
        private static int iterate(double cx, double cy, int max_iter)
        {
            double x = 0.0, y = 0.0, xx;
            int it;
            for (it = 0; (it < max_iter) && (x * x + y * y < 2 * 2); it++)
            {
                xx = x * x - y * y + cx;
                y = 2.0 * x * y + cy;
                x = xx;
            }
            return it;
        }


        public static void GetMandelbrotRegion(double x0, double x1, double y0, double y1, byte[] buffer, int res_x, int res_y, int max_iter, byte[] colors)
        {
            double mx = (x1 - x0) / res_x;
            double my = (y1 - y0) / res_y;
            int bi = 0;
            Parallel.For(0, res_y, delegate(int y)
            //for (int y = 0; y < res_y; y++)
            {
                for (int x = 0; x < res_x; x++)
                {
                    double cx = x0 + x * mx;
                    double cy = y0 + y * my;
                    int v = iterate(cx, cy, max_iter);

                    //(y * res_x + x) * 3 + 0

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

            byte r = (byte)linear(bas, 0, range[i], colors[i * 3 + 0], colors[(i + 1) * 3 + 0]);
            byte g = (byte)linear(bas, 0, range[i], colors[i * 3 + 1], colors[(i + 1) * 3 + 1]);
            byte b = (byte)linear(bas, 0, range[i], colors[i * 3 + 2], colors[(i + 1) * 3 + 2]);
            return new byte[] { r, g, b };
        }
        static public double linear(double x, double x0, double x1, double y0, double y1)
        {
            if ((x1 - x0) == 0)
            {
                return (y0 + y1) / 2;
            }
            return y0 + (x - x0) * (y1 - y0) / (x1 - x0);
        }
    }
}
