using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Reflection;

namespace MathTools
{
    public class Function
    {
        public static double DerivateInc = 1.0 / 2048.0;
        static string bat_top = @"public static double bat_top(MyDouble x) {double res = 2 * Math.Sqrt(-Math.Abs(Math.Abs(x) - 1) * Math.Abs(3 - Math.Abs(x)) / ((Math.Abs(x) - 1) * (3 - Math.Abs(x)))) * (1 + Math.Abs(Math.Abs(x) - 3) / (Math.Abs(x) - 3)) * Math.Sqrt(1 - Math.Pow((x / 7.0), 2)) + (5 + 0.97 * (Math.Abs(x - 0.5) + Math.Abs(x + .5)) - 3 * (Math.Abs(x - .75) + Math.Abs(x + .75))) * (1 + Math.Abs(1 - Math.Abs(x)) / (1 - Math.Abs(x)));
                    if (double.IsNaN(res))
                        res = (2.71052 + (1.5 - 0.5 * Math.Abs(x)) - 1.35526 * Math.Sqrt(4 - Math.Pow((Math.Abs(x) - 1), 2))) * Math.Sqrt(Math.Abs(Math.Abs(x) - 1) / (Math.Abs(x) - 1)) + 0.9;
                    return res;}";

        static string bat_bot = @"public static double bat_bot(MyDouble x)
        {
            double res = Math.Abs(x / 2.0) - 0.0913722 * (x * x) - 3.0 + Math.Sqrt(1 - Math.Pow((Math.Abs(Math.Abs(x) - 2) - 1), 2));
            if (double.IsNaN(res))
                res = -3 * Math.Sqrt(1 - Math.Pow((x / 7.0), 2)) * Math.Sqrt(Math.Abs(Math.Abs(x) - 4) / (Math.Abs(x) - 4));
            return res;
        }";
        public string Name
        {
            get
            {
                return _name;
            }
        }

        private CodeCompiler _function;
        private string _f, _name;

        public Function(string name, string f)
        {
            _name = name;
            _f = Fix(f);
            SetOthers(null, null, new string[] { "x" });
        }
        public Function(string name, string f, string[] parameters)
        {
            _name = name;
            _f = Fix(f);
            SetOthers(null, null, parameters);
        }
        public Function(string name, string f, string[] others_name, string[] others_f)
        {
            _name = name;
            _f = Fix(f);
            if (f.Replace(name + "(", "") != f)
                throw new Exception("Recorsive function");

            SetOthers(others_name, others_f, new string[] { "x" });
        }




        static double inc = 1;
        public delegate double FunctionDelegate(MyDouble x);

        public static FunctionDelegate D(FunctionDelegate F, double g, double inc)
        {
            for (int i = 1; i < g; i++)
                F = D(F);

            FunctionDelegate fd = delegate (MyDouble x)
            {
                return (F(x + inc / 2.0) - F(x - inc / 2.0)) / inc;
            };
            return fd;
        }
        public static FunctionDelegate D(FunctionDelegate F, double g)
        {
            return D(F, g, inc);
        }
        public static FunctionDelegate D(FunctionDelegate F)
        {
            return D(F, 1, inc);
        }

        static List<FunctionDelegate> Int_Func = new List<FunctionDelegate>();
        static List<FunctionDelegate> Int_Func_F = new List<FunctionDelegate>();
        static List<double> Int_Func_from = new List<double>();
        static List<double> Int_Func_to = new List<double>();
        static List<double> Int_Func_dx = new List<double>();
        public static FunctionDelegate I(FunctionDelegate F, double from, double to, double dx)
        {
            dx = Math.Abs(dx);
            for (int i = 0; i < Int_Func.Count; i++)
            {
                if (Int_Func_F[i].Method == F.Method && Int_Func_from[i] == from && Int_Func_to[i] == to && Int_Func_dx[i] == dx)
                    return Int_Func[i];
            }

            List<double> vals_y = new List<double>((int)(Math.Abs(to - from) / dx));
            List<double> vals_x = new List<double>((int)(Math.Abs(to - from) / dx));

            double a = 0;
            if (from < to)
                for (double f = from; f <= to; f += dx)
                {
                    vals_y.Add(a);
                    vals_x.Add(f);
                    a += dx * F(f);
                }
            else
            {
                for (double f = from; f >= to; f -= dx)
                {
                    vals_y.Add(a);
                    vals_x.Add(f);
                    a += dx * F(f);
                }
                vals_x.Reverse();
                vals_y.Reverse();
            }

            FunctionDelegate fd = delegate (MyDouble x)
            {
                double max = vals_x[vals_x.Count - 1] - vals_x[0];
                x = x - vals_x[0];
                if (x < 0)
                    return double.NaN;
                if (x > max)
                    return double.NaN;
                return vals_y[(int)((x / max) * (double)(vals_x.Count - 1))];
            };
            Int_Func.Add(fd);
            Int_Func_F.Add(F);
            Int_Func_from.Add(from);
            Int_Func_to.Add(to);
            Int_Func_dx.Add(dx);
            return fd;
        }
        public static FunctionDelegate I(FunctionDelegate F, double from, double to)
        {
            return I(F, from, to, inc);
        }



        public void SetOthers(string[] others_name, string[] others_f, string[] parameters)
        {
            string others_s = "";
            for (int i = 0; others_name != null && i < others_name.Length; i++)
            {
                if (others_name[i] != _name)
                {
                    others_s += "public static MyDouble " + others_name[i] + "(MyDouble x){return " + Fix(others_f[i]) + ";} ";
                    if (others_f[i].Replace(_name + "(", "") != others_f[i] && _f.Replace(others_name[i] + "(", "") != _f)
                        throw new Exception("Recorsive functions");
                }
            }// (Ys[i] - Ys[i + 1]) / (Xs[i] - Xs[i + 1]);
            string param = "";
            for (int i = 0; i < parameters.Length; i++)
            {
                param += "MyDouble " + parameters[i] + ",";
            }
            param = param.Substring(0, param.Length - 1);

            _function = new CodeCompiler(@"public class F{
                                          public static MyDouble pi = Math.PI;
                                          public static MyDouble e = Math.E;

                                          /****Metodi per la derivazione dinamica****/
                                          static double inc = Function.DerivateInc;
                                          public delegate MyDouble FunctionDelegate(MyDouble x);                                          
                                          public static FunctionDelegate D(FunctionDelegate F, double g, double inc)
                                          {
                                               for (int i = 1; i < g; i++)
                                                   F = D(F);
                                               
                                               FunctionDelegate fd = delegate(MyDouble x)
                                               {
                                                   return (F(x + inc/2.0) - F(x-inc/2.0))/inc;
                                               };
                                               return fd;
                                          }
                                          public static FunctionDelegate D(FunctionDelegate F, double g)
                                          {
                                               return D(F,g,inc);
                                          }
                                          public static FunctionDelegate D(FunctionDelegate F)
                                          {
                                               return D(F,1,inc);
                                          }


                                           static List<FunctionDelegate> Int_Func = new List<FunctionDelegate>();
                                            static List<FunctionDelegate> Int_Func_F = new List<FunctionDelegate>();
                                            static List<double> Int_Func_from = new List<double>();
                                            static List<double> Int_Func_to = new List<double>();
                                            static List<double> Int_Func_dx = new List<double>();
                                            public static FunctionDelegate I(FunctionDelegate F, double from, double to, double dx)
                                            {
                                                 dx = Math.Abs(dx);
            for (int i = 0; i < Int_Func.Count; i++)
            {
                if (Int_Func_F[i].Method == F.Method && Int_Func_from[i] == from && Int_Func_to[i] == to && Int_Func_dx[i] == dx)
                    return Int_Func[i];
            }

            List<double> vals_y = new List<double>((int)(Math.Abs(to - from) / dx));
            List<double> vals_x = new List<double>((int)(Math.Abs(to - from) / dx));

            double a = 0;
            if(from<to)
                for (double f = from; f <= to; f += dx)
                {
                    vals_y.Add(a);
                    vals_x.Add(f);
                    a += dx * F(f);   
                }
            else
            {
                for (double f = from; f >= to; f -= dx)
                {
                    vals_y.Add(-a);
                    vals_x.Add(f);
                    a += dx * F(f);
                }
                vals_x.Reverse();
                vals_y.Reverse();
            }


                                                FunctionDelegate fd = delegate(MyDouble x)
                                                {
                                                    double max = vals_x[vals_x.Count - 1] - vals_x[0];
                                                    x = x - vals_x[0];
                                                    if (x < 0)
                                                        return double.NaN;
                                                    if (x > max)
                                                        return double.NaN;
                                                    return vals_y[(int)((x / max) * (double)(vals_x.Count - 1))];
                                                };
                                                Int_Func.Add(fd);
                                                Int_Func_F.Add(F);
                                                Int_Func_from.Add(from);
                                                Int_Func_to.Add(to);
                                                Int_Func_dx.Add(dx);
                                                return fd;
                                            }
public static FunctionDelegate I(FunctionDelegate F, double from=0, double to=20)
        {
            return I(F, from, to, inc);
        }

                                        public static MyDouble sin(MyDouble x){return Math.Sin(x);}
                                        public static MyDouble cos(MyDouble x){return Math.Cos(x);}
                                        public static MyDouble tan(MyDouble x){return Math.Tan(x);}
                                        public static MyDouble atan(MyDouble x){return Math.Atan(x);}
                                        public static MyDouble acos(MyDouble x){return Math.Acos(x);}
                                        public static MyDouble asin(MyDouble x){return Math.Asin(x);}
                                        public static MyDouble sinh(MyDouble x){return Math.Sinh(x);}
                                        public static MyDouble cosh(MyDouble x){return Math.Cosh(x);}
                                        public static MyDouble tanh(MyDouble x){return Math.Tanh(x);}
                                        public static MyDouble abs(MyDouble x){return Math.Abs(x);}

                                        public static MyDouble log(MyDouble x){return Math.Log(x,10.0);}
                                        public static MyDouble log(MyDouble x, MyDouble b){return Math.Log(x,b);}
                                        public static MyDouble ln(MyDouble x){return Math.Log(x);}
                                        public static MyDouble pow(MyDouble x, MyDouble ex){return Math.Pow(x,ex);}
                                        public static MyDouble sqrt(MyDouble x){return Math.Sqrt(x);}

                                        public static MyDouble remainder(MyDouble x, MyDouble p){return Math.IEEERemainder(x,p);}
                                        public static MyDouble sign(MyDouble x){return Math.Sign(x);}
                                        public static MyDouble floor(MyDouble x){return Math.Floor(x);}
                                        public static MyDouble ceiling(MyDouble x){return Math.Ceiling(x);}

                                        public static MyDouble module(MyDouble x,MyDouble d){return x - d * ((MyDouble)((int)(x/d))); }

                                          public static MyDouble " + _name + "("+ param + "){return " + _f + ";}"
                                        + others_s +
                                          bat_top +
                                          bat_bot + "}"

                                         , _name, "using System; using MathTools; using System.Collections.Generic;", false, "Mandelbrot_Generator.exe");
        }
        /*Ritorna un singolo valore della funzione*/
        public double Value(double x)
        {
            MyDouble a = (MyDouble)_function.InvokeStatic("F", _name, new MyDouble(x));
            return (double)a;
        }
        public double Value(params object[] parameters)
        {
            for (int i = 0; i < parameters.Length; i++) parameters[i] = new MyDouble((double)parameters[i]);
            MyDouble a = (MyDouble)_function.InvokeStatic("F", _name, parameters);
            return (double)a;
        }
        /*Ritorna un insieme di valori della funzione, con le x associate*/
        public double[] Values(double start, double end, double inc, out double[] xs)
        {
            MethodInfo m = _function.GetMethodFromClass("F", _name);
            List<double> ly = new List<double>((int)Math.Abs((end - start) / inc));
            List<double> lx = new List<double>((int)Math.Abs((end - start) / inc));

            while (start <= end)
            {
                ly.Add((MyDouble)m.Invoke(null, new object[] { new MyDouble(start) }));
                lx.Add(start);
                start += inc;
            }
            xs = lx.ToArray();
            return ly.ToArray();
        }
        public double[] DerivateValues(double start, double end, double inc, out double[] xs)
        {
            MethodInfo m = _function.GetMethodFromClass("F", _name);
            List<double> ly = new List<double>((int)Math.Abs((end - start) / inc));
            List<double> lx = new List<double>((int)Math.Abs((end - start) / inc));

            while (start <= end)
            {
                ly.Add((double)m.Invoke(null, new object[] { new MyDouble(start) }));
                lx.Add(start);
                start += inc;
            }
            xs = lx.ToArray();
            return Derivate(xs, ly.ToArray());
        }



        /*Metodi per il calcolo differenziale*/
        /*Calcola la derivata per un certo dominio di una funzione*/
        public static double[] Derivate(double[] Xs, double[] Ys)
        {
            double[] YsDerivates = new double[Ys.Length];
            for (int i = 0; i < Ys.Length - 1; i++)
            {
                YsDerivates[i] = (Ys[i] - Ys[i + 1]) / (Xs[i] - Xs[i + 1]);
            }
            if (YsDerivates.Length >= 2)
                YsDerivates[YsDerivates.Length - 1] = YsDerivates[YsDerivates.Length - 2];
            return YsDerivates;
        }

        public static double[] Integrate(double[] Xs, double[] Ys, double inc, double startX, double endX)
        {
            double[] YsIntegral = new double[Ys.Length];
            int k0 = (int)(Math.Abs(startX) / inc);
            if (startX >= 0)
                k0 = 0;
            if (startX < 0 && endX < 0)
                k0 = Xs.Length - 1;

            double y = 0;
            for (int i = k0; i < Ys.Length; i++)
            {
                if (!double.IsInfinity(Ys[i]) && !double.IsNaN(Ys[i]))
                {
                    y += Ys[i] * inc;
                    YsIntegral[i] = y;
                }
                else
                    YsIntegral[i] = double.NaN;
            }
            y = 0;
            for (int i = k0; i >= 0; i--)
            {
                if (!double.IsInfinity(Ys[i]) && !double.IsNaN(Ys[i]))
                {
                    y -= Ys[i] * inc;
                    YsIntegral[i] = y;
                }
                else
                    YsIntegral[i] = double.NaN;
            }
            return YsIntegral;
        }

        public static double[] Integrate(double[] Xs, double[] Ys, double inc, double from, double to, out double[] NewXs)
        {
            List<double> nYs = new List<double>(1000), nXs = new List<double>(1000);
            int k0 = 0;
            while (Xs[k0] < from) k0++;
            double y = 0;
            while (Xs[k0] <= to)
            {
                k0++;
                if (!double.IsInfinity(Ys[k0]) && !double.IsNaN(Ys[k0]))
                {
                    y += Ys[k0] * inc;
                    nYs.Add(y);
                    nXs.Add(Xs[k0]);
                }
                else
                {
                    nYs.Add(double.NaN);
                    nXs.Add(Xs[k0]);
                }
            }
            NewXs = nXs.ToArray();
            return nYs.ToArray();
        }







        /*Metodi per fixare la funzione in modo da tradurla il linguaggio c#*/
        public static string Fix(string f)
        {
            f = FixNumbers(FixNames(f)).Replace("^", "%");
            bool open = true;
            for (int i = 0; i < f.Length; i++)
            {
                if (f[i] == '|' && open)
                {
                    f = f.Remove(i, 1);
                    f = f.Insert(i, "abs(");
                    open = !open;
                    i = 0;
                }
                else if (f[i] == '|' && !open)
                {
                    f = f.Remove(i, 1);
                    f = f.Insert(i, ")");
                    open = !open;
                    i = 0;
                }

            }
            return f;
        }
        /*Sistemo i nomi  sin(x) => Math.Sin(x) */
        private static string[] to_sub = new string[] { "remainder", "sign", "floor", "tan", "aMath.Tan", "ceiling", "cos", "aMath.Cos", "sin", "aMath.Sin", "pow", "sqrt", "ln", "log", "abs" };
        private static string[] sub = new string[] { "Math.IEEERemainder", "Math.Sign", "Math.Floor", "Math.Tan", "Math.Atan", "Math.Ceiling", "Math.Cos", "Math.Acos", "Math.Sin", "Math.Asin", "Math.Pow", "Math.Sqrt", "Math.Log", "Math.Log", "Math.Abs" };
        private static string FixNames(string f)
        {
            /*
            for (int i = 0; i < to_sub.Length; i++)
            {
                f = f.Replace(to_sub[i], sub[i]);
            }*/
            return f;
        }
        /*Sistemo i numeri   4/5 => 4.0/5.0 */
        private static char[] numbers = new char[] { '0', '1', '2', '3', '4', '5', '6', '7', '8', '9' };
        private static string FixNumbers(string f)
        {
            bool n = false;
            for (int i = 0; i < f.Length; i++)
            {
                if (numbers.Any(c => c == f[i]) && !n)
                {
                    n = true;
                    f = f.Insert(i, "(MyDouble)");
                    i += "(MyDouble)".Length;
                }
                else if (f[i] == '.' && n)
                {
                    n = false;
                    i++;
                    while (i < f.Length && numbers.Any(c => c == f[i]))
                        i++;
                }
                else if (numbers.All(c => c != f[i]) && n)
                {
                    n = false;
                    if (f[i] != '.')
                        f = f.Insert(i, ".0");
                    i += 2;
                }
            }
            if (n == true)
            {
                f = f.Insert(f.Length, ".0");
            }

            return f;
        }
    }
    public struct MyDouble
    {
        double value;
        public MyDouble(double value)
        {
            this.value = value;
        }


        #region Unary Negation operator
        public static MyDouble operator -(MyDouble left)
        {
            return new MyDouble(-left.value);
        }
        public static MyDouble operator !(MyDouble left)
        {
            return new MyDouble(Fatt(left));
        }
        #endregion

        #region Addition operators
        public static MyDouble operator +(MyDouble left, MyDouble right)
        {
            return new MyDouble(left.value + right.value);
        }
        public static MyDouble operator +(double left, MyDouble right)
        {
            return new MyDouble(left + right.value);
        }
        public static MyDouble operator +(MyDouble left, double right)
        {
            return new MyDouble(left.value + right);
        }

        #endregion

        #region Subtraction operators
        public static MyDouble operator -(MyDouble left, MyDouble right)
        {
            return new MyDouble(left.value - right.value);
        }
        public static MyDouble operator -(double left, MyDouble right)
        {
            return new MyDouble(left - right.value);
        }
        public static MyDouble operator -(MyDouble left, double right)
        {
            return new MyDouble(left.value - right);
        }
        #endregion

        #region Multiplication operators
        public static MyDouble operator *(MyDouble left, MyDouble right)
        {
            return new MyDouble(left.value * right.value);
        }
        public static MyDouble operator *(double left, MyDouble right)
        {
            return new MyDouble(left * right.value);
        }
        public static MyDouble operator *(MyDouble left, double right)
        {
            return new MyDouble(left.value * right);
        }
        #endregion

        #region Division operators
        public static MyDouble operator /(MyDouble left, MyDouble right)
        {
            return new MyDouble(left.value / right.value);
        }
        public static MyDouble operator /(double left, MyDouble right)
        {
            return new MyDouble(left / right.value);
        }
        public static MyDouble operator /(MyDouble left, double right)
        {
            return new MyDouble(left.value / right);
        }
        #endregion

        #region Modulus operators
        /*
        public static MyDouble operator %(MyDouble left, MyDouble right)
        {
            return new MyDouble(left.value % right.value);
        }
        public static MyDouble operator %(double left, MyDouble right)
        {
            return new MyDouble(left % right.value);
        }
        public static MyDouble operator %(MyDouble left, double right)
        {
            return new MyDouble(left.value % right);
        }*/
        #endregion

        #region Power operators
        public static MyDouble operator %(MyDouble left, MyDouble right)
        {
            return new MyDouble(Math.Pow(left.value, right.value));
        }
        public static MyDouble operator %(double left, MyDouble right)
        {
            return new MyDouble(Math.Pow(left, right.value));
        }
        public static MyDouble operator %(MyDouble left, double right)
        {
            return new MyDouble(Math.Pow(left.value, right));
        }
        #endregion




        public static MyDouble Fatt(MyDouble x)
        {
            if (x == 0)
                return 0;

            int m = Math.Sign(x.value);
            x = Math.Abs(x.value);
            double r = 1;
            for (int i = 1; i <= x.value; i++)
            {
                r *= i;
            }
            return r * m;
        }

        #region Implict conversion from primitive operators
        public static implicit operator MyDouble(double value)
        {
            return new MyDouble(value);
        }
        public static implicit operator MyDouble(long value)
        {
            return new MyDouble(value);
        }
        public static implicit operator double(MyDouble value)
        {
            return value.value;
        }
        #endregion

        #region Explicit converstion to primitive operators

        public static implicit operator string(MyDouble frac)
        {
            return frac.value.ToString();
        }

        #endregion

        public override string ToString()
        {
            return value.ToString();
        }
    }
}
