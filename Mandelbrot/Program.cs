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
        static byte[] colors = new byte[(maxit + 1) * 3];
        static void Main(string[] args)
        {

            for (int i = 0; i < maxit + 1; i++)
            {
                Color c = GetLinearGradient(i, 0, maxit,
                    new Color[] { Color.DarkGray, Color.DarkGray, Color.Black, Color.DarkRed, Color.Red, Color.DarkRed, Color.Black },
                    new double[] { 1,1,3,3,3,0.5 });
                colors[i * 3 + 0] = c.R;
                colors[i * 3 + 1] = c.G;
                colors[i * 3 + 2] = c.B;
            }

            int resy = 2000;
            int resx = 2000;
            string file = "lol.ppm";


            if (File.Exists(file)) File.Delete(file);
            BinaryWriter writer = new BinaryWriter(File.OpenWrite(file), Encoding.Default);


            writer.Write(("P6\n" + resx + " " + resy + "\n255\n").ToCharArray().Select(c => (byte)c).ToArray());


            byte[] buff = new byte[resy * resx * 3];

            Mandelbrot.GetMandelbrotRegion(-3, 3, -3, 3, buff, resx, resy, maxit, colors);
                writer.Write(buff);
            



            /*
            for (int y = 0; y < resy; y++)
            {
                Console.WriteLine("drawing " + y + " line");
                
            }*/


            writer.Close();
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
