/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership.  The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
package org.apache.beam.sdk.extensions.sketching;

import java.io.Serializable;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

import org.apache.avro.Schema;
import org.apache.avro.SchemaBuilder;
import org.apache.avro.generic.GenericData;
import org.apache.avro.generic.GenericRecord;
import org.apache.beam.sdk.coders.AvroCoder;
import org.apache.beam.sdk.coders.Coder;
import org.apache.beam.sdk.coders.VarIntCoder;
import org.apache.beam.sdk.coders.VarLongCoder;
import org.apache.beam.sdk.extensions.sketching.SketchFrequencies.CountMinSketchFn;
import org.apache.beam.sdk.extensions.sketching.SketchFrequencies.Sketch;
import org.apache.beam.sdk.testing.CoderProperties;
import org.apache.beam.sdk.testing.PAssert;
import org.apache.beam.sdk.testing.TestPipeline;
import org.apache.beam.sdk.transforms.Combine;
import org.apache.beam.sdk.transforms.Create;
import org.apache.beam.sdk.transforms.Distinct;
import org.apache.beam.sdk.transforms.DoFn;
import org.apache.beam.sdk.transforms.ParDo;
import org.apache.beam.sdk.transforms.SerializableFunction;
import org.apache.beam.sdk.transforms.Values;
import org.apache.beam.sdk.transforms.View;
import org.apache.beam.sdk.transforms.WithKeys;
import org.apache.beam.sdk.values.KV;
import org.apache.beam.sdk.values.PCollection;
import org.apache.beam.sdk.values.PCollectionView;
import org.junit.Assert;
import org.junit.Rule;
import org.junit.Test;

/**
 * Tests for {@link SketchFrequencies}.
 */
public class SketchFrequenciesTest implements Serializable {

  @Rule public final transient TestPipeline tp = TestPipeline.create();

  private List<Long> smallStream = Arrays.asList(
          1L,
          2L, 2L,
          3L, 3L, 3L,
          4L, 4L, 4L, 4L,
          5L, 5L, 5L, 5L, 5L,
          6L, 6L, 6L, 6L, 6L, 6L,
          7L, 7L, 7L, 7L, 7L, 7L, 7L,
          8L, 8L, 8L, 8L, 8L, 8L, 8L, 8L,
          9L, 9L, 9L, 9L, 9L, 9L, 9L, 9L, 9L,
          10L, 10L, 10L, 10L, 10L, 10L, 10L, 10L, 10L, 10L);

  private Long[] distinctElems = {1L, 2L, 3L, 4L, 5L, 6L, 8L, 9L, 10L};
  private Long[] frequencies = distinctElems.clone();

  @Test
  public void globallyDefault() {
    PCollection<Long> stream = tp.apply(Create.of(smallStream));
    PCollection<Sketch<Long>> sketch = stream.apply(SketchFrequencies
            .<Long>globally());

    Coder<Long> coder = stream.getCoder();
    PAssert.thatSingleton("Verify number of hits", sketch)
              .satisfies(new VerifyStreamFrequencies<Long>(coder, distinctElems, frequencies));

    tp.run();
  }

  @Test
  public void perKeyDefault() {
    PCollection<Long> stream = tp.apply(Create.of(smallStream));
    PCollection<Sketch<Long>> sketch = stream
            .apply(WithKeys.<Integer, Long>of(1))
            .apply(SketchFrequencies.<Integer, Long>perKey())
            .apply(Values.<Sketch<Long>>create());

    Coder<Long> coder = stream.getCoder();

    PAssert.thatSingleton("Verify number of hits", sketch)
            .satisfies(new VerifyStreamFrequencies<Long>(coder, distinctElems, frequencies));

    tp.run();
  }

  @Test
  public void parameterTuning() {
    double eps = 0.01;
    double conf = 0.8;
    PCollection<Long> stream = tp.apply(Create.of(smallStream));
    PCollection<Sketch<Long>> sketch = stream
            .apply(SketchFrequencies
                    .<Long>globally()
                    .withRelativeError(eps)
                    .withConfidence(conf));

    Coder<Long> coder = stream.getCoder();

    PAssert.thatSingleton("Verify number of hits", sketch)
            .satisfies(new VerifyStreamFrequencies<Long>(coder, distinctElems, frequencies));

    tp.run();
  }

  @Test
  public void useCombineFn() {
    SketchFrequencies.CountMinSketchFn<Long> cmsFn = SketchFrequencies.CountMinSketchFn
            .<Long>create(VarLongCoder.of());
    PCollection<Long> stream = tp.apply(Create.of(smallStream));
    PCollection<Sketch<Long>> sketch = stream
            .apply(Combine.<Long, Sketch<Long>>globally(cmsFn));

    Coder<Long> coder = stream.getCoder();

    PAssert.thatSingleton("Verify number of hits", sketch)
            .satisfies(new VerifyStreamFrequencies<Long>(coder, distinctElems, frequencies));


    tp.run();
  }

  @Test
  public void merge() {
    double eps = 0.01;
    double conf = 0.9;

    List<Sketch<Integer>> sketches = new ArrayList<>();
    Coder<Integer> coder = VarIntCoder.of();

    // 3 sketches respectively containing : [0, 1, 2], [1, 2, 3] and [2, 3, 4]
    for (int i = 0; i < 3; i++) {
      sketches.add(new Sketch<Integer>(eps, conf));
      for (int j = 0; j < 5; j++) {
        sketches.get(i).add(j, coder);
      }
    }

    CountMinSketchFn<Integer> fn = CountMinSketchFn.create(coder).withAccuracy(eps, conf);
    Sketch<Integer> merged = fn.mergeAccumulators(sketches);
    for (int i = 0; i < 5; i++) {
      Assert.assertEquals(3, merged.estimateCount(i, coder));
    }
  }

  @Test
  public void query() {
    PCollection<Long> stream = tp.apply(Create.of(smallStream));
    PCollection<Sketch<Long>> sketch = stream.apply(SketchFrequencies
            .<Long>globally());

    final Coder<Long> coder = stream.getCoder();

    // build a view of the Count-Min sketch so it can be passed as sideInput.
    final PCollectionView<Sketch<Long>> sketchView = sketch.apply(View
            .<Sketch<Long>>asSingleton());

    PCollection<KV<Long, Long>> pairs = stream.apply(ParDo.of(
            new DoFn<Long, KV<Long, Long>>() {
              @ProcessElement
              public void procesElement(ProcessContext c) {
                Long elem = c.element();
                Sketch<Long> sketch = c.sideInput(sketchView);
                sketch.estimateCount(elem, coder);
              }}).withSideInputs(sketchView));

    PAssert.that(pairs).satisfies(new SerializableFunction<Iterable<KV<Long, Long>>, Void>() {
      @Override
      public Void apply(Iterable<KV<Long, Long>> input) {
        for (KV<Long, Long> pair : input) {
          Assert.assertEquals(KV.of(pair.getKey(), pair.getKey()), pair);
        }
        return null;
      }
    });
    tp.run();

  }

  @Test
  public void customObject() {
    Schema schema =
            SchemaBuilder.record("User")
                    .fields()
                    .requiredString("Pseudo")
                    .requiredLong("Expected_frequency")
                    .endRecord();
    final List<GenericRecord> users = new ArrayList<>();
    for (long i = 1L; i < 11L; i++) {
      GenericData.Record newRecord = new GenericData.Record(schema);
      newRecord.put("Pseudo", "User" + i);
      newRecord.put("Expected_frequency", i);
      // Add i times the new record : the frequency is equal to the age
      for (int j = 0; j < i; j++) {
        users.add(newRecord);
      }
    }

    final AvroCoder<GenericRecord> coder = AvroCoder.of(schema);
    PCollection<GenericRecord> stream = tp.apply("Create stream",
            Create.of(users).withCoder(coder));
    PCollection<Sketch<GenericRecord>> sketch = stream.apply("Test custom object",
            SketchFrequencies.<GenericRecord>globally());

    final PCollectionView<Sketch<GenericRecord>> sketchView = sketch.apply("Get sketch View",
            View.<Sketch<GenericRecord>>asSingleton());
    PCollection<GenericRecord> uniqueUsers = stream.apply("Get Distinct Records",
            Distinct.<GenericRecord>create());

    uniqueUsers.apply("Verify number of hits", ParDo.of(new DoFn<GenericRecord, Void>() {
      @ProcessElement
      public void processElement(ProcessContext c) {
        GenericRecord user = c.element();
        Sketch<GenericRecord> sketch = c.sideInput(sketchView);
        // The frequency of each record is equal to the "Age" of the user
        Assert.assertEquals(user.get("Expected_frequency"),
                sketch.estimateCount(user, coder));
      }}).withSideInputs(sketchView));
    tp.run();
  }

  @Test
  public void testCoder() throws Exception {
    Sketch<Long> cMSketch = new Sketch<Long>(0.01, 0.9);
    Coder<Long> coder = VarLongCoder.of();
    for (long i = 0L; i < 10L; i++) {
      cMSketch.add(i, coder);
    }

    CoderProperties.<Sketch<Long>>coderDecodeEncodeEqual(
            new SketchFrequencies.CountMinSketchCoder<Long>(), cMSketch);
  }

  static class VerifyStreamFrequencies<T> implements SerializableFunction<Sketch<T>, Void> {

    Coder<T> coder;
    Long[] expectedHits;
    T[] elements;

    VerifyStreamFrequencies(Coder<T> coder, T[] elements, Long[] expectedHits) {
      this.coder = coder;
      this.elements = elements;
      this.expectedHits = expectedHits;
    }

    @Override
    public Void apply(Sketch<T> sketch) {
      for (int i = 0; i < elements.length; i++) {
        Assert.assertEquals((long) expectedHits[i], sketch.estimateCount(elements[i], coder));
      }
      return null;
    }
  }
}
