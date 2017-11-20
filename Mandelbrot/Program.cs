using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.IO;
using System.Drawing;

namespace Mandelbrot
{
    class Program
    {
        static int maxit = 100;
        static Color[] colors = new Color[maxit + 1];
        static void Main(string[] args)
        {

            for (int i = 0; i < maxit + 1; i++)
            {
                colors[i] = GetLinearGradient(i, 0, maxit,
                    new Color[] { Color.DarkGray, Color.DarkGray, Color.Black, Color.DarkRed, Color.Red, Color.DarkRed, Color.Black },
                    new double[] { 0.6,1,3,3,3,0.5 });
            }

            int resy = 2000;
            int resx = (int)(resy * 16.0/9.0);
            string file = "lol.ppm";


            if (File.Exists(file)) File.Delete(file);
            BinaryWriter writer = new BinaryWriter(File.OpenWrite(file), Encoding.Default);


            writer.Write(("P6\n" + resx + " " + resy + "\n255\n").ToCharArray().Select(c => (byte)c).ToArray());

            int cunch = 10;
            byte[] buff = new byte[resy / cunch * resx * 3];
            for (int i = 0; i < cunch; i++)
            {
                draw_lines(resy / cunch * i, resy / cunch * (i + 1), buff, resx, resy);
                writer.Write(buff);
            }



            /*
            for (int y = 0; y < resy; y++)
            {
                Console.WriteLine("drawing " + y + " line");
                
            }*/


            writer.Close();
        }
        static int iterate(double cx, double cy)
        {
            double x = 0.0, y = 0.0, xx;
            int it;
            for (it = 0; (it < maxit) && (x * x + y * y < 2 * 2); it++)
            {
                xx = x * x - y * y + cx;
                y = 2.0 * x * y + cy;
                x = xx;
            }
            return it;
        }

        
        static void draw_lines(int ystart, int yend, byte[] buffer, int xsize, int ysize)
        {

            Parallel.For(ystart, yend, delegate(int y)
            //for (int y = ystart; y < yend; y++)
            {
                if (y % 100 == 0)
                    Console.WriteLine("drawing " + y + " line");
                for (int x = 0; x < xsize; x++)
                {
                    double cx = -2.5 + 5 * ((double)x) / (xsize - 1);
                    double cy = 1.2 - 2.4 * ((double)y) / (ysize - 1);
                    int v = iterate(cx, cy);

                    buffer[((y - ystart) * xsize + x) * 3 + 0] = (byte)(colors[v].R);
                    buffer[((y - ystart) * xsize + x) * 3 + 1] = (byte)(colors[v].G);
                    buffer[((y - ystart) * xsize + x) * 3 + 2] = (byte)(colors[v].B);
                }
            });
        }

        static Color GetLinearGradient(double v, double minV, double maxV, Color[] colors, double[] range)
        {
            int i = 0;

            double sum = range.Sum();

            double bas = (v - minV) / (maxV - minV) * sum;

            while (i < range.Length - 1 && bas > range[i] )
            {
                bas -= range[i];
                i++;
            }

            byte r = (byte)linear(bas, 0, range[i], colors[i].R, colors[i + 1].R);
            byte g = (byte)linear(bas, 0, range[i], colors[i].G, colors[i + 1].G);
            byte b = (byte)linear(bas, 0, range[i], colors[i].B, colors[i + 1].B);
            return Color.FromArgb(r, g, b);
        }

        static Color GetLinearGradient(double v, double minV, double maxV, params Color[] colors)
        {
            double step = (maxV - minV) / (colors.Length - 1);
            int i = (int)(v / step);
            if (i >= colors.Length - 1) i = colors.Length - 2;

            v = v - step * i;

            byte r = (byte)linear(v, 0, step, colors[i].R, colors[i + 1].R);
            byte g = (byte)linear(v, 0, step, colors[i].G, colors[i + 1].G);
            byte b = (byte)linear(v, 0, step, colors[i].B, colors[i + 1].B);
            return Color.FromArgb(r, g, b);
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
