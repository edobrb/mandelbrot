using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using Microsoft.CSharp;
using System.CodeDom.Compiler;
using System.Reflection;

namespace MathTools
{
    public class CodeCompiler
    {
        Assembly _assembly;
        string _namespace_name;
        public CodeCompiler(string code, string namespace_name, string usings, bool generate_assembly, params string[] references)
        {
            _namespace_name = namespace_name;
            ReloadCode(code, namespace_name, usings, generate_assembly, references);
        }
        private void ReloadCode(string code, string namespace_name, string usings, bool generate_assembly, params string[] references)
        {
            code = usings + " namespace " + namespace_name + " { " + code + " }";


            CSharpCodeProvider provider = new CSharpCodeProvider();
            CompilerParameters parameters = new CompilerParameters();
            parameters.ReferencedAssemblies.AddRange(references);
            parameters.GenerateInMemory = true;
            parameters.GenerateExecutable = false;
            if (generate_assembly)
                parameters.OutputAssembly = namespace_name;

            CompilerResults results = provider.CompileAssemblyFromSource(parameters, code);
            if (results.Errors.HasErrors)
            {
                StringBuilder sb = new StringBuilder();
                foreach (CompilerError error in results.Errors)
                {
                    sb.AppendLine(String.Format("Error ({0}): {1}", error.ErrorNumber, error.ErrorText));
                }
                throw new Exception(sb.ToString());

            }

            _assembly = results.CompiledAssembly;
        }


        public Type GetClass(string name)
        {
            return _assembly.GetType(_namespace_name + "." + name);
        }
        public object CreateClassIstance(Type class_type, params object[] prms)
        {
            return Activator.CreateInstance(class_type, prms);
        }
        public object CreateClassIstance(string class_name, params object[] prms)
        {
            return Activator.CreateInstance(GetClass(class_name), prms);
        }

        public MethodInfo GetMethodFromClass(Type class_type, string method_name)
        {
            return class_type.GetMethod(method_name);
        }
        public MethodInfo GetMethodFromClass(string class_name, string method_name)
        {
            return GetClass(class_name).GetMethod(method_name);
        }


        public object InvokeStatic(string class_name, string method_name, params object[] prms)
        {
            return GetMethodFromClass(class_name, method_name).Invoke(null, prms);
        }
        public object InvokeDinamic(string class_name, string method_name, object class_istance, params object[] prms)
        {
            return GetMethodFromClass(class_name, method_name).Invoke(class_istance, prms);
        }
    }
}
