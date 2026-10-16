using System;
using Cavern;
using Cavern.Format;
using Cavern.Format.Common;
using Cavern.Format.Environment;
using Cavern.Format.Renderers;
using Cavern.Format.Utilities;

static class P {
    static int Main(string[] a) {
        if (a.Length < 2) { Console.WriteLine("usage: ec3_to_adm in.ec3 out.wav"); return 1; }
        AudioReader reader = AudioReader.Open(a[0]);
        reader.ReadHeader();
        Renderer r = reader.GetRenderer();
        Console.WriteLine("channels={0} len={1} sr={2} bits={3} hasObjects={4} objects={5} bedChannels={6}",
            reader.ChannelCount, reader.Length, reader.SampleRate, reader.Bits, r.HasObjects, r.Objects.Count, r.Channels);
        reader.Reset();
        Listener listener = new Listener();
        listener.SampleRate = reader.SampleRate;
        EnvironmentWriter w = EnvironmentWriter.Create(a[1], Codec.ADM_BWF, listener, reader.Length, BitDepth.Int24);
        Transcoder.Transcode(reader, w);
        w.Dispose();
        reader.Dispose();
        Console.WriteLine("done");
        return 0;
    }
}
